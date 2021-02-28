import os
os.system('cd /home/karam/PycharmProjects/cat-dog-classification')
os.system('python search-bing-api.py --query "cat" --output images')
os.system('python search-bing-api.py --query "dog" --output images')
os.system('python mymodel.py')