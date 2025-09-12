"""
Production Monitoring Utilities for Homodyne v2
===============================================

Comprehensive production monitoring and alerting system for scientific
computing workflows in production environments. Features:

- Real-time health monitoring with configurable metrics
- Performance baseline comparison and drift detection
- Alert system for critical issues with multiple notification channels
- Resource utilization monitoring and capacity planning
- Analysis pipeline health checks and validation
- Automated anomaly detection and escalation procedures
- Integration with common monitoring platforms (Prometheus, Grafana)
- Scientific workflow-specific monitoring contexts

This module provides production-grade monitoring capabilities while
maintaining integration with the existing homodyne logging infrastructure.
"""

import functools
import json
import logging
import os
import socket
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from collections import defaultdict, deque
from enum import Enum
import sqlite3
import hashlib

# System monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Scientific computing monitoring
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None

# HTTP/email alerts
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None

try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    HAS_EMAIL = True
except ImportError:
    HAS_EMAIL = False

from .logging import get_logger
from .advanced_debugging import get_advanced_debugging_stats
from .distributed_logging import get_distributed_computing_stats


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics that can be monitored."""
    COUNTER = "counter"           # Always increasing
    GAUGE = "gauge"              # Can go up or down
    HISTOGRAM = "histogram"       # Distribution of values
    TIMER = "timer"              # Duration measurements


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    check_name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    execution_time: float
    timestamp: float = field(default_factory=time.time)
    details: Optional[Dict[str, Any]] = None
    recommendations: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """Alert notification structure."""
    level: AlertLevel
    title: str
    message: str
    source: str  # Component that generated the alert
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp
    
    @property
    def alert_id(self) -> str:
        """Generate unique alert ID based on content."""
        content = f"{self.title}_{self.source}_{self.message}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class Metric:
    """Individual metric measurement."""
    name: str
    value: Union[float, int]
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    operation_name: str
    baseline_duration: float
    baseline_std: float
    sample_count: int
    creation_time: float
    confidence_interval: Tuple[float, float]  # 95% CI
    percentiles: Dict[int, float]  # 50th, 90th, 95th, 99th percentiles
    
    def is_anomalous(self, duration: float, threshold_factor: float = 2.0) -> bool:
        """Check if a duration is anomalous compared to baseline."""
        threshold = self.baseline_duration + (threshold_factor * self.baseline_std)
        return duration > threshold


class MetricsCollector:
    """Collect and aggregate metrics from various sources."""
    
    def __init__(self, logger: Optional = None):
        self.logger = logger or get_logger(__name__)
        self._metrics = deque(maxlen=10000)  # Keep recent metrics
        self._aggregations = defaultdict(list)  # For computing aggregates
        self._lock = threading.RLock()
        
        # Start background aggregation thread
        self._stop_aggregation = threading.Event()
        self._aggregation_thread = threading.Thread(
            target=self._background_aggregation, 
            daemon=True
        )
        self._aggregation_thread.start()
    
    def record_metric(self, metric: Metric):
        """Record a metric measurement."""
        with self._lock:
            self._metrics.append(metric)
            self._aggregations[metric.name].append(metric.value)
            
            # Keep aggregation lists bounded
            if len(self._aggregations[metric.name]) > 1000:
                self._aggregations[metric.name] = self._aggregations[metric.name][-500:]
    
    def record_counter(self, name: str, value: Union[float, int] = 1, 
                      tags: Optional[Dict[str, str]] = None, unit: Optional[str] = None):
        """Record a counter metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags or {},
            unit=unit
        )
        self.record_metric(metric)
    
    def record_gauge(self, name: str, value: Union[float, int],
                    tags: Optional[Dict[str, str]] = None, unit: Optional[str] = None):
        """Record a gauge metric."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags or {},
            unit=unit
        )
        self.record_metric(metric)
    
    def record_timer(self, name: str, duration: float,
                    tags: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        metric = Metric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            tags=tags or {},
            unit="seconds"
        )
        self.record_metric(metric)
    
    def get_metric_summary(self, metric_name: str, 
                          window_minutes: int = 60) -> Optional[Dict[str, Any]]:
        """Get summary statistics for a metric within time window."""
        cutoff_time = time.time() - (window_minutes * 60)
        
        with self._lock:
            # Filter metrics by name and time window
            relevant_metrics = [
                m for m in self._metrics 
                if m.name == metric_name and m.timestamp >= cutoff_time
            ]
            
            if not relevant_metrics:
                return None
            
            values = [m.value for m in relevant_metrics]
            
            summary = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'latest': values[-1],
                'window_minutes': window_minutes
            }
            
            # Add percentiles if numpy available
            if HAS_NUMPY and len(values) > 1:
                values_array = np.array(values)
                summary.update({
                    'std': float(np.std(values_array)),
                    'median': float(np.median(values_array)),
                    'p90': float(np.percentile(values_array, 90)),
                    'p95': float(np.percentile(values_array, 95)),
                    'p99': float(np.percentile(values_array, 99))
                })
            
            return summary
    
    def _background_aggregation(self):
        """Background thread for metric aggregation."""
        while not self._stop_aggregation.wait(60):  # Run every minute
            try:
                self._compute_aggregates()
            except Exception as e:
                self.logger.error(f"Background metric aggregation failed: {e}")
    
    def _compute_aggregates(self):
        """Compute and log metric aggregates."""
        with self._lock:
            for metric_name, values in self._aggregations.items():
                if len(values) > 0:
                    summary = {
                        'metric': metric_name,
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values)
                    }
                    
                    self.logger.debug(f"Metric aggregate: {summary}")
    
    def stop(self):
        """Stop the metrics collector."""
        self._stop_aggregation.set()
        if self._aggregation_thread.is_alive():
            self._aggregation_thread.join(timeout=5)


class HealthChecker:
    """Perform various health checks on the system and analysis pipeline."""
    
    def __init__(self, logger: Optional = None):
        self.logger = logger or get_logger(__name__)
        self._health_checks = {}
        self._results_history = deque(maxlen=1000)
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("memory_usage", self._check_memory_usage)
        
        if HAS_JAX:
            self.register_check("jax_devices", self._check_jax_devices)
        
        self.register_check("log_directory", self._check_log_directory)
    
    def register_check(self, check_name: str, check_function: Callable[[], HealthCheckResult]):
        """Register a custom health check."""
        self._health_checks[check_name] = check_function
    
    def run_check(self, check_name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if check_name not in self._health_checks:
            return HealthCheckResult(
                check_name=check_name,
                status="unhealthy",
                message="Health check not found",
                execution_time=0.0
            )
        
        start_time = time.perf_counter()
        
        try:
            result = self._health_checks[check_name]()
            result.execution_time = time.perf_counter() - start_time
            
        except Exception as e:
            result = HealthCheckResult(
                check_name=check_name,
                status="unhealthy", 
                message=f"Health check failed: {e}",
                execution_time=time.perf_counter() - start_time
            )
        
        self._results_history.append(result)
        return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        for check_name in self._health_checks:
            results[check_name] = self.run_check(check_name)
        
        return results
    
    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource availability."""
        if not HAS_PSUTIL:
            return HealthCheckResult(
                check_name="system_resources",
                status="degraded",
                message="psutil not available for system monitoring",
                execution_time=0.0
            )
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            status = "healthy"
            issues = []
            
            if cpu_percent > 90:
                status = "degraded" if cpu_percent < 95 else "unhealthy"
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > 85:
                status = "degraded" if memory.percent < 95 else "unhealthy"
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            message = "System resources OK" if status == "healthy" else "; ".join(issues)
            
            return HealthCheckResult(
                check_name="system_resources",
                status=status,
                message=message,
                execution_time=0.0,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'available_memory_gb': memory.available / 1024**3
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name="system_resources",
                status="unhealthy",
                message=f"Failed to check system resources: {e}",
                execution_time=0.0
            )
    
    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability."""
        if not HAS_PSUTIL:
            return HealthCheckResult(
                check_name="disk_space",
                status="degraded",
                message="psutil not available for disk monitoring",
                execution_time=0.0
            )
        
        try:
            # Check root partition
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            free_gb = disk.free / 1024**3
            
            status = "healthy"
            if usage_percent > 85:
                status = "degraded" if usage_percent < 95 else "unhealthy"
            
            message = f"Disk usage: {usage_percent:.1f}% ({free_gb:.1f}GB free)"
            
            recommendations = []
            if status != "healthy":
                recommendations.append("Clean up temporary files and old logs")
                recommendations.append("Archive or compress old data files")
            
            return HealthCheckResult(
                check_name="disk_space",
                status=status,
                message=message,
                execution_time=0.0,
                details={
                    'usage_percent': usage_percent,
                    'free_gb': free_gb,
                    'total_gb': disk.total / 1024**3
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name="disk_space",
                status="unhealthy",
                message=f"Failed to check disk space: {e}",
                execution_time=0.0
            )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage patterns."""
        if not HAS_PSUTIL:
            return HealthCheckResult(
                check_name="memory_usage",
                status="degraded",
                message="psutil not available for memory monitoring",
                execution_time=0.0
            )
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            status = "healthy"
            recommendations = []
            
            # Check for potential memory leaks (process using > 1GB)
            memory_mb = memory_info.rss / 1024**2
            if memory_mb > 1024:
                status = "degraded" if memory_mb < 2048 else "unhealthy"
                recommendations.append("Monitor for potential memory leaks")
                recommendations.append("Consider garbage collection or restart")
            
            message = f"Process memory: {memory_mb:.0f}MB ({memory_percent:.1f}% of system)"
            
            return HealthCheckResult(
                check_name="memory_usage",
                status=status,
                message=message,
                execution_time=0.0,
                details={
                    'memory_mb': memory_mb,
                    'memory_percent': memory_percent,
                    'vms_mb': memory_info.vms / 1024**2
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name="memory_usage",
                status="unhealthy",
                message=f"Failed to check memory usage: {e}",
                execution_time=0.0
            )
    
    def _check_jax_devices(self) -> HealthCheckResult:
        """Check JAX device availability."""
        if not HAS_JAX:
            return HealthCheckResult(
                check_name="jax_devices",
                status="degraded",
                message="JAX not available",
                execution_time=0.0
            )
        
        try:
            devices = jax.devices()
            device_info = [str(d) for d in devices]
            
            gpu_count = sum(1 for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower())
            
            status = "healthy"
            message = f"JAX devices: {len(devices)} total"
            if gpu_count > 0:
                message += f" ({gpu_count} GPU)"
            
            return HealthCheckResult(
                check_name="jax_devices",
                status=status,
                message=message,
                execution_time=0.0,
                details={
                    'device_count': len(devices),
                    'gpu_count': gpu_count,
                    'devices': device_info
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                check_name="jax_devices",
                status="unhealthy",
                message=f"Failed to check JAX devices: {e}",
                execution_time=0.0
            )
    
    def _check_log_directory(self) -> HealthCheckResult:
        """Check log directory accessibility."""
        try:
            log_dir = Path.home() / '.homodyne' / 'logs'
            
            if not log_dir.exists():
                return HealthCheckResult(
                    check_name="log_directory",
                    status="degraded",
                    message="Log directory does not exist",
                    execution_time=0.0,
                    recommendations=["Create log directory or check logging configuration"]
                )
            
            # Check write permissions
            test_file = log_dir / f"health_check_{int(time.time())}.tmp"
            try:
                test_file.write_text("health check")
                test_file.unlink()
                
                return HealthCheckResult(
                    check_name="log_directory",
                    status="healthy",
                    message=f"Log directory accessible: {log_dir}",
                    execution_time=0.0
                )
                
            except Exception:
                return HealthCheckResult(
                    check_name="log_directory",
                    status="unhealthy",
                    message="Log directory not writable",
                    execution_time=0.0,
                    recommendations=["Check directory permissions"]
                )
                
        except Exception as e:
            return HealthCheckResult(
                check_name="log_directory",
                status="unhealthy",
                message=f"Failed to check log directory: {e}",
                execution_time=0.0
            )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        if not self._results_history:
            return {'status': 'no_data'}
        
        # Get latest results for each check
        latest_results = {}
        for result in reversed(self._results_history):
            if result.check_name not in latest_results:
                latest_results[result.check_name] = result
        
        # Overall status
        statuses = [r.status for r in latest_results.values()]
        if 'unhealthy' in statuses:
            overall_status = 'unhealthy'
        elif 'degraded' in statuses:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        return {
            'overall_status': overall_status,
            'check_count': len(latest_results),
            'healthy_checks': sum(1 for s in statuses if s == 'healthy'),
            'degraded_checks': sum(1 for s in statuses if s == 'degraded'),
            'unhealthy_checks': sum(1 for s in statuses if s == 'unhealthy'),
            'individual_results': {name: asdict(result) for name, result in latest_results.items()}
        }


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, logger: Optional = None):
        self.logger = logger or get_logger(__name__)
        self._active_alerts = {}  # alert_id -> Alert
        self._alert_history = deque(maxlen=1000)
        self._notification_handlers = {}
        self._lock = threading.RLock()
        
        # Default notification settings
        self.notification_settings = {
            'email_enabled': False,
            'webhook_enabled': False,
            'log_all_levels': True,
            'min_level_for_email': AlertLevel.ERROR,
            'min_level_for_webhook': AlertLevel.WARNING
        }
    
    def configure_email_notifications(self, smtp_host: str, smtp_port: int,
                                    username: str, password: str, 
                                    recipients: List[str]):
        """Configure email notifications."""
        if not HAS_EMAIL:
            self.logger.warning("Email support not available")
            return
        
        self.notification_settings.update({
            'email_enabled': True,
            'smtp_host': smtp_host,
            'smtp_port': smtp_port,
            'smtp_username': username,
            'smtp_password': password,
            'email_recipients': recipients
        })
    
    def configure_webhook_notifications(self, webhook_url: str, 
                                      headers: Optional[Dict[str, str]] = None):
        """Configure webhook notifications."""
        if not HAS_REQUESTS:
            self.logger.warning("Webhook support not available")
            return
        
        self.notification_settings.update({
            'webhook_enabled': True,
            'webhook_url': webhook_url,
            'webhook_headers': headers or {}
        })
    
    def raise_alert(self, level: AlertLevel, title: str, message: str,
                   source: str, tags: Optional[Dict[str, str]] = None) -> Alert:
        """Raise a new alert."""
        alert = Alert(
            level=level,
            title=title,
            message=message,
            source=source,
            tags=tags or {}
        )
        
        with self._lock:
            self._active_alerts[alert.alert_id] = alert
            self._alert_history.append(alert)
        
        # Send notifications
        self._send_notifications(alert)
        
        # Log the alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(level, logging.INFO)
        
        self.logger.log(log_level, f"ALERT [{level.value.upper()}] {title}: {message} (source: {source})")
        
        return alert
    
    def resolve_alert(self, alert_id: str, resolution_message: Optional[str] = None):
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = time.time()
                
                if resolution_message:
                    alert.message += f" [RESOLVED: {resolution_message}]"
                
                del self._active_alerts[alert_id]
                
                self.logger.info(f"Alert resolved: {alert.title} (ID: {alert_id})")
                
                return alert
        
        return None
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications via configured channels."""
        
        # Email notification
        if (self.notification_settings.get('email_enabled') and 
            alert.level.value >= self.notification_settings['min_level_for_email'].value):
            
            threading.Thread(
                target=self._send_email_notification, 
                args=(alert,), 
                daemon=True
            ).start()
        
        # Webhook notification
        if (self.notification_settings.get('webhook_enabled') and
            alert.level.value >= self.notification_settings['min_level_for_webhook'].value):
            
            threading.Thread(
                target=self._send_webhook_notification,
                args=(alert,),
                daemon=True
            ).start()
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification for alert."""
        if not HAS_EMAIL:
            return
        
        try:
            settings = self.notification_settings
            
            msg = MIMEMultipart()
            msg['From'] = settings['smtp_username']
            msg['To'] = ', '.join(settings['email_recipients'])
            msg['Subject'] = f"[Homodyne Alert - {alert.level.value.upper()}] {alert.title}"
            
            body = f"""
Alert Details:
- Level: {alert.level.value.upper()}
- Title: {alert.title}
- Message: {alert.message}
- Source: {alert.source}
- Time: {datetime.fromtimestamp(alert.timestamp).isoformat()}
- Alert ID: {alert.alert_id}

Tags: {json.dumps(alert.tags, indent=2)}

This alert was generated by the Homodyne production monitoring system.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(settings['smtp_host'], settings['smtp_port'])
            server.starttls()
            server.login(settings['smtp_username'], settings['smtp_password'])
            
            text = msg.as_string()
            server.sendmail(settings['smtp_username'], settings['email_recipients'], text)
            server.quit()
            
            self.logger.debug(f"Email notification sent for alert: {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
    
    def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification for alert."""
        if not HAS_REQUESTS:
            return
        
        try:
            payload = {
                'alert_id': alert.alert_id,
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'timestamp': alert.timestamp,
                'tags': alert.tags
            }
            
            response = requests.post(
                self.notification_settings['webhook_url'],
                json=payload,
                headers=self.notification_settings['webhook_headers'],
                timeout=10
            )
            
            response.raise_for_status()
            self.logger.debug(f"Webhook notification sent for alert: {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts."""
        with self._lock:
            return list(self._active_alerts.values())
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        with self._lock:
            active_alerts = list(self._active_alerts.values())
            
            # Count by level
            level_counts = defaultdict(int)
            for alert in active_alerts:
                level_counts[alert.level.value] += 1
            
            # Recent alert count (last hour)
            cutoff_time = time.time() - 3600
            recent_alerts = [a for a in self._alert_history if a.timestamp > cutoff_time]
            
            return {
                'active_alert_count': len(active_alerts),
                'recent_alert_count_1h': len(recent_alerts),
                'alert_level_counts': dict(level_counts),
                'oldest_active_alert_age': max([a.age_seconds for a in active_alerts]) if active_alerts else 0,
                'total_alerts_history': len(self._alert_history)
            }


# Global instances
_metrics_collector = MetricsCollector()
_health_checker = HealthChecker()
_alert_manager = AlertManager()


def monitor_performance(operation_name: str, 
                       baseline_duration: Optional[float] = None,
                       alert_on_anomaly: bool = True):
    """Decorator for monitoring operation performance."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time
                
                # Record metrics
                _metrics_collector.record_timer(f"operation.{operation_name}.duration", duration)
                _metrics_collector.record_counter(f"operation.{operation_name}.success")
                
                # Check for performance anomalies
                if baseline_duration and alert_on_anomaly:
                    if duration > baseline_duration * 3:  # 3x slower than baseline
                        _alert_manager.raise_alert(
                            level=AlertLevel.WARNING,
                            title=f"Performance Anomaly: {operation_name}",
                            message=f"Operation took {duration:.2f}s (baseline: {baseline_duration:.2f}s)",
                            source=f"{func.__module__}.{func.__qualname__}",
                            tags={'operation': operation_name, 'duration': str(duration)}
                        )
                
                return result
                
            except Exception as e:
                duration = time.perf_counter() - start_time
                
                # Record failure metrics
                _metrics_collector.record_counter(f"operation.{operation_name}.failure")
                _metrics_collector.record_timer(f"operation.{operation_name}.failure_duration", duration)
                
                # Raise error alert
                _alert_manager.raise_alert(
                    level=AlertLevel.ERROR,
                    title=f"Operation Failed: {operation_name}",
                    message=f"Operation failed after {duration:.2f}s: {str(e)}",
                    source=f"{func.__module__}.{func.__qualname__}",
                    tags={'operation': operation_name, 'error_type': type(e).__name__}
                )
                
                raise
        
        return wrapper
    return decorator


@contextmanager
def production_monitoring_context(operation_name: str,
                                critical: bool = False,
                                expected_duration: Optional[float] = None):
    """Context manager for production monitoring of operations."""
    logger = get_logger()
    start_time = time.perf_counter()
    
    # Record operation start
    _metrics_collector.record_counter(f"operation.{operation_name}.started")
    
    logger.info(f"Production operation started: {operation_name}")
    
    try:
        yield
        
        # Success
        duration = time.perf_counter() - start_time
        _metrics_collector.record_timer(f"operation.{operation_name}.duration", duration)
        _metrics_collector.record_counter(f"operation.{operation_name}.success")
        
        # Check expected duration
        if expected_duration and duration > expected_duration * 2:
            alert_level = AlertLevel.CRITICAL if critical else AlertLevel.WARNING
            _alert_manager.raise_alert(
                level=alert_level,
                title=f"Operation Duration Alert: {operation_name}",
                message=f"Operation took {duration:.2f}s (expected: {expected_duration:.2f}s)",
                source="production_monitoring",
                tags={'operation': operation_name}
            )
        
        logger.info(f"Production operation completed: {operation_name} [{duration:.2f}s]")
        
    except Exception as e:
        # Failure
        duration = time.perf_counter() - start_time
        _metrics_collector.record_counter(f"operation.{operation_name}.failure")
        
        alert_level = AlertLevel.CRITICAL if critical else AlertLevel.ERROR
        _alert_manager.raise_alert(
            level=alert_level,
            title=f"Critical Operation Failed: {operation_name}" if critical else f"Operation Failed: {operation_name}",
            message=f"Operation failed after {duration:.2f}s: {str(e)}",
            source="production_monitoring",
            tags={'operation': operation_name, 'error_type': type(e).__name__}
        )
        
        logger.error(f"Production operation failed: {operation_name} [{duration:.2f}s]: {e}")
        raise


def run_health_checks() -> Dict[str, Any]:
    """Run all health checks and return results."""
    return _health_checker.run_all_checks()


def get_production_monitoring_stats() -> Dict[str, Any]:
    """Get comprehensive production monitoring statistics."""
    return {
        'health_summary': _health_checker.get_health_summary(),
        'alert_summary': _alert_manager.get_alert_summary(),
        'active_alerts': [asdict(alert) for alert in _alert_manager.get_active_alerts()],
        'system_capabilities': {
            'has_psutil': HAS_PSUTIL,
            'has_requests': HAS_REQUESTS,
            'has_email': HAS_EMAIL,
            'has_numpy': HAS_NUMPY,
            'has_jax': HAS_JAX
        },
        'monitoring_status': {
            'metrics_collected': len(_metrics_collector._metrics),
            'health_checks_run': len(_health_checker._results_history),
            'alerts_total': len(_alert_manager._alert_history)
        }
    }


def export_monitoring_dashboard(filepath: Optional[Path] = None) -> str:
    """Export monitoring data for dashboard visualization."""
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = Path(f"homodyne_monitoring_dashboard_{timestamp}.json")
    
    dashboard_data = {
        'generated_at': datetime.now().isoformat(),
        'hostname': socket.gethostname(),
        'production_stats': get_production_monitoring_stats(),
        'advanced_debugging_stats': get_advanced_debugging_stats(),
        'distributed_computing_stats': get_distributed_computing_stats()
    }
    
    with open(filepath, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    
    logger = get_logger(__name__)
    logger.info(f"Monitoring dashboard exported to: {filepath}")
    
    return str(filepath)


# Cleanup function
def shutdown_production_monitoring():
    """Shutdown production monitoring components."""
    _metrics_collector.stop()
    logger = get_logger(__name__)
    logger.info("Production monitoring shutdown complete")