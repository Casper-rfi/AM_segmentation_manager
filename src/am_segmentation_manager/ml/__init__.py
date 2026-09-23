"""Optional inference support: turning images into predictions with a model.

Everything under here needs torch, which nothing outside it may import, and
nothing here is imported unless a prediction is being made; :mod:`.environment`
reports what is missing rather than failing on import. ``preprocessing.py`` is a
copy of AM_training's module, a precursor of the training package the models are
now trained with; only its training-only transform was removed, so a diff against
that source shows that removal and nothing else. Code needing different behaviour
wraps it. Generation 1 runs it, held to Adaptive Milling's own answers by
``tests/test_ml_generation.py``.
"""
