"""Optional inference and training support: predicting with a model, and training one.

Everything under here needs torch, which nothing outside it may import, and
nothing here is imported unless a prediction or a training run is being made;
:mod:`.environment` reports what is missing rather than failing on import.
``preprocessing.py`` is generation 1's preprocessing, held to Adaptive Milling's
own answers by ``tests/test_ml_generation.py``; ``trainer.py`` is the one place
AM-model-training is imported.
"""
