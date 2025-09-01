{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:

Quick Reference
---------------

{% if attributes %}

.. rubric:: Attributes

{% for item in attributes %}
{% if item not in ['at',] %}
* :attr:`~{{ objname }}.{{ item }}`
{% endif %}
{% endfor %}
{% endif %}

{% if methods %}

.. rubric:: Methods

{% for item in methods %}
{% if item not in ['__init__',] %}
* :attr:`~{{ objname }}.{{ item }}`
{% endif %}
{% endfor %}
{% endif %}



{% if attributes %}
Attributes
==========

{% for item in attributes %}
{% if item not in ['at',] %}
.. autoattribute:: {{ objname }}.{{ item }}
{% endif %}
{% endfor %}

{% endif %}



{% if methods %}
Methods
==========

{% for item in methods %}

{% if item not in ['__init__',] %}
.. automethod:: {{ objname }}.{{ item }}
{% endif %}

{% endfor %}
{% endif %}


If you find any errors in the documentation, please report them in the `Github Issues <https://github.com/ymahlau/fdtdx/issues>`_!
