cd package
python -m build 
cd dist 
pip uninstall package
pip install package-0.1-py3-none-any.whl --force-reinstall
cd ../..