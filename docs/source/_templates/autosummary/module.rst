{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
   :noindex:

{% if modules %}
Submodules
----------

.. autosummary::
   :toctree:
   :recursive:

{% for item in modules %}
   {{ item }}
{% endfor %}
{% endif %}
