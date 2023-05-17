//============================================train model==================================//
1- to run app.py (used to test training model which trained for all data set):
	- go to date folder and put all test images 
	- run file app.py
	- get results in folder output 
		-results.txt: for each test case
		-time.txt: time taken for each test case
//=========================================to try all the flow of code===============================//
2-to run pipeline of all data set (used to get training model for all data set and save model in models folder with name"soft/hard_training_voting"): 	
	- go to file Resizer_training.py :
		- put path of all data folder in calling of (load_images) function in main below
		-put path of result of resized images folder in calling of (save_images) function in main below
	-go to file all_dataset_training_model.py:
		- go to file helper.py and put path of resized images folder in (load_all_dataset__win function)
   		- run all_dataset_training_model.py
3-to run pipeline (used to get traing model for split all data in train , test , valid and save model in models folder with name "soft/harf_voting")
	- go to file Resizer_training.py :
		- put path of all data folder in calling of (load_images) function in main below
		-put path of result of resized images folder in calling of (save_images) function in main below
	-go to file pipeline.py:
		- go to file helper.py and put path of resized images folder in (load_all_dataset__win function)
   		- run pipeline.py
   