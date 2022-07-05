.. Top level package

User API
========

This page only shows partial APIs relevant to users. See all APIs in :ref:`Developer API
</modules.rst>`.

NodeGAM Packages
-------------------------------------

.. autoclass:: nodegam.sklearn.NodeGAMClassifier
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: nodegam.sklearn.NodeGAMRegressor
    :members:
    :undoc-members:
    :show-inheritance:

Utilities
--------------
.. autofunction:: nodegam.utils.average_GAM_dfs
.. autofunction:: nodegam.utils.output_csv
.. autofunction:: nodegam.vis_utils.vis_GAM_effects


EBM Packages
-------------------------------------

.. autoclass:: nodegam.gams.MyEBM.MyExplainableBoostingClassifier
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: nodegam.gams.MyEBM.MyExplainableBoostingRegressor
    :members:
    :undoc-members:
    :show-inheritance:

Spline Packages
-------------------------------------

Note: Spline can be combined with :ref:`MyBagging </notebooks/tour_of_pygam.ipynb>` to get the
uncertainty.

.. autoclass:: nodegam.gams.MySpline.MySplineLogisticGAM
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: nodegam.gams.MySpline.MySplineGAM
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:

XGB-GAM Packages
-------------------------------------

Note: XGB-GAM can be combined with :ref:`MyBagging </notebooks/tour_of_pygam.ipynb>` to get the
uncertainty.

.. autoclass:: nodegam.gams.MyXGB.MyXGBOnehotClassifier
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: nodegam.gams.MyXGB.MyXGBOnehotRegressor
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:


Bagging Packages
-------------------------------------

.. autoclass:: nodegam.gams.MyBagging.MyBaggingClassifier
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: nodegam.gams.MyBagging.MyBaggingRegressor
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:




