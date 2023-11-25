
import os 
import glob
import pandas as pd

def collect_files(folder, contains, sort=True): 
    files = glob.glob((folder+ "*"))
    target_files = []

    for file in files:
        if contains in file:
            target_files.append(file)

    # if sort:
    #     target_files = target_files.sort() # Insure sorted 

    return target_files

def join_activity(folder):
    files = collect_files(folder, contains="ACTIVITY_LOGGER")
    accumulated_csv = pd.DataFrame()
    accumulated_time = 0

    for index, file in enumerate(files):
        files[index] = file.replace("\\", "/")
        print(files[index])

    filename = files[0].split("/")[-1]
    
    videoname = filename.split("_")[0]
    export_name =  folder + videoname + "_ACTIVITY_LOGGER_COMPILED.csv"


    for file in files:


        joining_csv = pd.read_csv(file)

        # assign accumulated_csv if empty
        joining_csv['Accumulated_Event_Time'] = joining_csv["Event_Time"] + accumulated_time
        # print(joining_csv["Event_Time"] + accumulated_time)
        # print(joining_csv['Accumulated_Event_Time'])

        # print(joining_csv)

        # Copies rows
        if accumulated_csv.empty:
            accumulated_csv = joining_csv.iloc[:0,:].copy()

        accumulated_csv = accumulated_csv.append(joining_csv)
        # print(accumulated_csv)

        # get last column of duration and add that for next time
        tracked_duration = joining_csv["Event_Time"].iloc[-1]
        
        accumulated_time += tracked_duration
        # print(accumulated_time, tracked_duration)
    print(accumulated_csv)
    accumulated_csv.to_csv(export_name, index=False)
    return accumulated_csv

def join_tracked(folder, video_name):
    files = collect_files(folder, contains = (video_name + "_S"))
    accumulated_csv = pd.DataFrame()

    for index, file in enumerate(files):
        files[index] = file.replace("\\", "/")
        print(files[index])

    filename = files[0].split("/")[-1]
    videoname = filename.split("_")[0]
    export_name =  folder + videoname + "_COMPILED.csv"

    print(files)
    for file in files:
        joining_csv = pd.read_csv(file)

        # Copies rows
        if accumulated_csv.empty:
            accumulated_csv = joining_csv.iloc[:0,:].copy()

        else:
            joining_csv = joining_csv.iloc[2:]

        accumulated_csv = accumulated_csv.append(joining_csv)

    accumulated_csv.to_csv(export_name, index=False)
    return accumulated_csv


if __name__ == "__main__":
    # join_activity(folder="K:/Github/PeopleTracker/Evaluation/People/Justin_Tracked/GP044104_Tracked/")
    # join_tracked(folder="K:/Github/PeopleTracker/Evaluation/People/Justin_Tracked/GP044104_Tracked/", video_name="GP044104")

    join_activity(folder="K:/Github/PeopleTracker/R_Statistics/R_data/Completed Evaluations-20230726T173343Z-001/Completed Evaluations/Sam/Galleries/Historical/")
    join_tracked(folder="K:/Github/PeopleTracker/R_Statistics/R_data/Completed Evaluations-20230726T173343Z-001/Completed Evaluations/Sam/Galleries/Historical/", video_name="GP054106")