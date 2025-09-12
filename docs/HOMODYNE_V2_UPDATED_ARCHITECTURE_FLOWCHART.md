# Homodyne v2 Updated Architecture Flow Chart

## Overview
This flow chart represents the current state of Homodyne v2 after comprehensive enhancements including JAX fallback systems, advanced data loading pipeline, and performance optimizations.

## System Architecture Flow

```mermaid
graph TB
    %% Entry Points
    CLI[CLI Entry Point<br/>homodyne.cli.main]
    API[Python API<br/>homodyne.api.high_level]
    
    %% Configuration Layer
    ConfigMgr[Configuration Manager<br/>homodyne.config.manager<br/>- YAML-first configuration<br/>- Runtime parameter override<br/>- Performance optimization]
    
    %% Analysis Pipeline
    Pipeline[Analysis Pipeline<br/>homodyne.workflows.pipeline<br/>- Orchestrates complete workflow<br/>- Error handling & recovery<br/>- Progress reporting]
    
    %% Data Loading System (Enhanced)
    DataLoader[Enhanced XPCS Data Loader<br/>homodyne.data.xpcs_loader<br/>- Config-based filtering<br/>- Performance engine integration<br/>- Memory management]
    
    ConfigFilter[Config-Based Filtering<br/>homodyne.data.filtering_utils<br/>- Q-range filtering<br/>- Phi angle filtering<br/>- Quality-based filtering]
    
    Preprocessing[Preprocessing Pipeline<br/>homodyne.data.preprocessing<br/>- 7-stage processing<br/>- Enhanced diagonal correction<br/>- Multiple normalization methods]
    
    QualityController[Quality Controller<br/>homodyne.data.quality_controller<br/>- Progressive quality control<br/>- Auto-repair capabilities<br/>- Quality reporting]
    
    PerformanceEngine[Performance Engine<br/>homodyne.data.performance_engine<br/>- Memory-mapped I/O<br/>- Intelligent chunking<br/>- Multi-level caching]
    
    MemoryManager[Memory Manager<br/>homodyne.data.memory_manager<br/>- Dynamic allocation<br/>- Pressure monitoring<br/>- Memory pool management]
    
    %% Core Computational Layer
    JAXBackend[JAX Backend<br/>homodyne.core.jax_backend<br/>- JIT-compiled operations<br/>- GPU/TPU acceleration<br/>- Intelligent fallback]
    
    NumPyGradients[NumPy Gradients<br/>homodyne.core.numpy_gradients<br/>- Production-grade fallback<br/>- Complex-step differentiation<br/>- Richardson extrapolation]
    
    Models[Physical Models<br/>homodyne.core.models<br/>- DiffusionModel<br/>- ShearModel<br/>- CombinedModel]
    
    %% Optimization Layer (Enhanced)
    VIJax[Variational Inference<br/>homodyne.optimization.variational<br/>- JAX-accelerated VI<br/>- NumPy fallback support<br/>- KL divergence minimization]
    
    MCMCJax[MCMC Sampling<br/>homodyne.optimization.mcmc<br/>- JAX-accelerated sampling<br/>- Fallback MCMC<br/>- Multi-chain support]
    
    HybridOpt[Hybrid Optimizer<br/>homodyne.optimization.hybrid<br/>- VI → MCMC pipeline<br/>- Fallback integration<br/>- Best of both worlds]
    
    %% Results Processing
    ResultsMgr[Results Manager<br/>homodyne.results.manager<br/>- Multi-format export<br/>- Validation & processing<br/>- Error handling]
    
    Plotting[Plotting System<br/>homodyne.plotting<br/>- Automated visualization<br/>- Method-specific plots<br/>- Export capabilities]
    
    %% Support Systems
    Logging[Enhanced Logging<br/>homodyne.utils.logging<br/>- Performance monitoring<br/>- Scientific logging<br/>- Advanced debugging]
    
    HPC[HPC Integration<br/>homodyne.hpc<br/>- PBS Professional<br/>- Distributed computing<br/>- Resource management]
    
    %% Flow Connections - Entry to Configuration
    CLI --> ConfigMgr
    API --> ConfigMgr
    
    %% Configuration to Pipeline
    ConfigMgr --> Pipeline
    
    %% Pipeline to Data Loading System
    Pipeline --> DataLoader
    
    %% Enhanced Data Loading Flow
    DataLoader --> PerformanceEngine
    DataLoader --> MemoryManager
    DataLoader --> ConfigFilter
    DataLoader --> Preprocessing
    DataLoader --> QualityController
    
    ConfigFilter --> Preprocessing
    Preprocessing --> QualityController
    
    %% Performance Components Integration
    PerformanceEngine --> MemoryManager
    MemoryManager --> DataLoader
    
    %% Data to Computational Core
    QualityController --> JAXBackend
    JAXBackend --> NumPyGradients
    JAXBackend --> Models
    
    %% Core to Optimization
    Models --> VIJax
    Models --> MCMCJax
    Models --> HybridOpt
    
    NumPyGradients --> VIJax
    NumPyGradients --> MCMCJax
    NumPyGradients --> HybridOpt
    
    %% Optimization to Results
    VIJax --> ResultsMgr
    MCMCJax --> ResultsMgr
    HybridOpt --> ResultsMgr
    
    %% Results to Output
    ResultsMgr --> Plotting
    
    %% Support System Integration
    Logging -.-> Pipeline
    Logging -.-> DataLoader
    Logging -.-> JAXBackend
    Logging -.-> VIJax
    Logging -.-> MCMCJax
    
    HPC -.-> Pipeline
    HPC -.-> VIJax
    HPC -.-> MCMCJax
    
    %% Styling
    classDef entryPoint fill:#e1f5fe
    classDef dataSystem fill:#f3e5f5
    classDef computational fill:#e8f5e8
    classDef optimization fill:#fff3e0
    classDef results fill:#fce4ec
    classDef support fill:#f1f8e9
    classDef enhancement fill:#e0f2f1
    
    class CLI,API entryPoint
    class ConfigMgr,DataLoader,ConfigFilter,Preprocessing,QualityController dataSystem
    class PerformanceEngine,MemoryManager enhancement
    class JAXBackend,NumPyGradients,Models computational
    class VIJax,MCMCJax,HybridOpt optimization
    class ResultsMgr,Plotting results
    class Logging,HPC support
```

## Key Enhancement Integrations

### 1. **Enhanced Data Loading Pipeline**
```mermaid
graph LR
    RawHDF5[Raw HDF5 Data] --> PerfEngine[Performance Engine]
    PerfEngine --> MemManager[Memory Manager]
    MemManager --> ConfigFilter[Config-Based Filtering]
    ConfigFilter --> Preprocessing[7-Stage Preprocessing]
    Preprocessing --> QualityCtrl[Quality Controller]
    QualityCtrl --> ValidatedData[Validated Data]
```

### 2. **JAX Fallback Architecture**
```mermaid
graph TD
    JAXBackend[JAX Backend] --> JAXCheck{JAX Available?}
    JAXCheck -->|Yes| JAXOps[JAX JIT Operations]
    JAXCheck -->|No| FallbackSystem[Fallback System]
    
    FallbackSystem --> NumPyGrad[NumPy Gradients]
    FallbackSystem --> NumPyOps[NumPy Operations]
    
    JAXOps --> Results[Results]
    NumPyGrad --> Results
    NumPyOps --> Results
```

### 3. **Performance Optimization Layer**
```mermaid
graph TB
    LargeData[Large Dataset >1GB] --> MemMap[Memory-Mapped I/O]
    MemMap --> Chunking[Intelligent Chunking]
    Chunking --> ParallelProc[Parallel Processing]
    ParallelProc --> MultiCache[Multi-Level Caching]
    MultiCache --> Prefetch[Smart Prefetching]
    Prefetch --> OptimizedData[Optimized Data Access]
```

## Analysis Modes & Parameter Mapping

```mermaid
graph TD
    ConfigMode{Analysis Mode} --> StaticIso[Static Isotropic<br/>3 params: D₀, α, D_offset]
    ConfigMode --> StaticAniso[Static Anisotropic<br/>3 params + angle filtering]
    ConfigMode --> LaminarFlow[Laminar Flow<br/>7 params: + γ̇₀, β, γ̇_offset, φ₀]
    
    StaticIso --> DiffModel[Diffusion Model Only]
    StaticAniso --> DiffModelFilt[Diffusion Model + Filtering]
    LaminarFlow --> CombinedModel[Combined Model<br/>Diffusion + Shear]
```

## Physical Model Implementation

```mermaid
graph TB
    PhysicsEq[g₂(φ,t₁,t₂) = offset + contrast × g₁²] --> G1Calc[g₁ Calculation]
    
    G1Calc --> DiffusionTerm[D(t) = D₀ t^α + D_offset]
    G1Calc --> ShearTerm[γ̇(t) = γ̇₀ t^β + γ̇_offset]
    
    DiffusionTerm --> G1Combined[Combined g₁]
    ShearTerm --> G1Combined
    
    G1Combined --> G2Scaled[g₂ = offset + contrast × g₁²]
    G2Scaled --> ChiSquared[χ² Minimization]
```

## Optimization Method Flow

```mermaid
graph TD
    Method{Selected Method} --> VI[Variational Inference]
    Method --> MCMC[MCMC Sampling]
    Method --> Hybrid[Hybrid VI→MCMC]
    
    VI --> VIResult[VI Result<br/>- Parameter estimates<br/>- Uncertainty quantification]
    
    MCMC --> MCMCResult[MCMC Result<br/>- Full posterior<br/>- Chain diagnostics]
    
    Hybrid --> VIInit[VI Initialization]
    VIInit --> MCMCRefine[MCMC Refinement]
    MCMCRefine --> HybridResult[Hybrid Result<br/>- Best of both methods]
```

## Configuration System Architecture

```mermaid
graph TB
    YAMLConfig[YAML Configuration] --> ConfigManager[Configuration Manager]
    CLIArgs[CLI Arguments] --> ConfigManager
    EnvVars[Environment Variables] --> ConfigManager
    
    ConfigManager --> ModeDetection[Analysis Mode Detection]
    ConfigManager --> ParamValidation[Parameter Validation]
    ConfigManager --> PerformanceSettings[Performance Settings]
    
    ModeDetection --> ParameterCount{Parameter Count}
    ParameterCount -->|3| StaticMode[Static Mode]
    ParameterCount -->|7| LaminarMode[Laminar Flow Mode]
```

## Data Quality Control Flow

```mermaid
graph TB
    RawData[Raw Data] --> Stage1[Stage 1: Basic Validation]
    Stage1 --> FilteredData[Filtered Data]
    FilteredData --> Stage2[Stage 2: Filter Validation]
    Stage2 --> PreprocessedData[Preprocessed Data]
    PreprocessedData --> Stage3[Stage 3: Transform Validation]
    Stage3 --> Stage4[Stage 4: Final Validation]
    Stage4 --> QualityReport[Quality Report<br/>Score: 0-100]
    Stage4 --> AutoRepair[Auto-Repair<br/>Conservative/Aggressive]
    AutoRepair --> ValidatedData[Production-Ready Data]
```

## Technology Stack Integration

### Core Technologies
- **JAX**: Primary computational backend (10-50x speedup)
- **NumPy**: Fallback computational backend
- **YAML**: Configuration format
- **HDF5**: Data storage format
- **PyTorch/NumPyro**: Alternative MCMC backend

### Performance Technologies
- **JIT Compilation**: JAX just-in-time compilation
- **GPU/TPU**: Hardware acceleration when available
- **Memory Mapping**: Efficient large file access
- **Multi-threading**: Parallel data processing

### Quality Assurance
- **Progressive Validation**: Multi-stage quality control
- **Auto-Repair**: Intelligent data correction
- **Fallback Systems**: Graceful degradation
- **Comprehensive Testing**: 95%+ code coverage

## Summary

This updated flow chart reflects the current Homodyne v2 architecture with all recent enhancements:

1. **Enhanced Data Loading**: Config-based filtering, preprocessing pipeline, quality control
2. **Performance Optimization**: Memory management, intelligent caching, parallel processing
3. **JAX Fallback System**: Complete NumPy fallback for environments without JAX
4. **Quality Assurance**: Progressive validation and auto-repair capabilities
5. **Modern Architecture**: Modular, extensible, and maintainable design

The system now provides production-ready XPCS analysis with automatic adaptation to different computational environments while maintaining scientific accuracy and user-friendly operation.