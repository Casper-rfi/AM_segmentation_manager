"""Optional inference support: turning images into predictions with a model.

Everything under here needs torch, which nothing outside it may import, and
nothing here is imported unless a prediction is being made; :mod:`.environment`
reports what is missing rather than failing on import. ``preprocessing.py`` is
generation 1's preprocessing, held to Adaptive Milling's own answers by
``tests/test_ml_generation.py``.
"""
