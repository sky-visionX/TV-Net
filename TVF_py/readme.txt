cpython:build
python sliding_window_setup.py build_ext --inplace
python -m cProfile -o profile.stats ./main.py