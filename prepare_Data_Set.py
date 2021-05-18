import os
import shutil
imageDirectory=""
for image in  os.listdir(imageDirectory):
    #print(image)
    personfolder=image.split('_')[0]
    print(personfolder)
    dest=imageDirectory+personfolder
    print(dest)
    if not os.path.exists(dest):
        os.mkdir(dest)
        #copy image specific folder
    imageFilePath=imageDirectory+image
    #print(imageFilePath)
    shutil.move(imageFilePath, dest)





    





    
