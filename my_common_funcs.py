def get_binary_testset(dataset_name):
    """
    get_binary_testset: dataset_name -> train_pth, test_pth
    
    `DariusAf_Deepfake_Database` (train_test)
    `Celeb-avg-30-(train/test)`
    `Celeb-rnd-30-(train/test)`
    `Celeb-diff-30-(train/test)`
    """
    testset = None
    path_2_root = "../.."
    if dataset_name == "DariusAf_Deepfake_Database":
        testset = f"{path_2_root}/_DATASETS/DariusAf_Deepfake_Database/train_test"
    elif dataset_name == "Celeb-avg-30-test":
        testset = f"{path_2_root}/_DATASETS/Celeb-DF-v2/Celeb-avg-30-test"
    elif dataset_name == "Celeb-rnd-30-test":
        testset = f"{path_2_root}/_DATASETS/Celeb-DF-v2/Celeb-rnd-30-test"
    elif dataset_name == "Celeb-diff-30-test":
        testset = f"{path_2_root}/_DATASETS/Celeb-DF-v2/Celeb-diff-30-test"

    elif dataset_name == "Celeb-avg-30-train":
        testset = f"{path_2_root}/_DATASETS/Celeb-DF-v2/Celeb-avg-30"
    elif dataset_name == "Celeb-rnd-30-train":
        testset = f"{path_2_root}/_DATASETS/Celeb-DF-v2/Celeb-rnd-30"
    elif dataset_name == "Celeb-diff-30-train":
        testset = f"{path_2_root}/_DATASETS/Celeb-DF-v2/Celeb-diff-30"

    elif dataset_name == "DariusAf-OC": # unary
        testset = f"{path_2_root}/_DATASETS/DariusAf_Deepfake_Database-OC/real-train/"
    elif dataset_name == "DariusAf-OC-test": # binary
        testset = f"{path_2_root}/_DATASETS/DariusAf_Deepfake_Database-OC/realfake-test/"

    elif dataset_name == "Celeb-DF-v2-OC": # unary
        testset = f"{path_2_root}/_DATASETS/Celeb-DF-v2-OC/Celeb-rnd-30-OC-real-train/"
    elif dataset_name == "Celeb-DF-v2-OC-val": # unary, only has real class
        testset = f"{path_2_root}/_DATASETS/Celeb-DF-v2-OC/Celeb-rnd-30-OC-real-val/"
    elif dataset_name == "Celeb-DF-v2-OC-test": # unary
        testset = f"{path_2_root}/_DATASETS/Celeb-DF-v2-OC/Celeb-rnd-30-OC-realfake-test/"
    return testset