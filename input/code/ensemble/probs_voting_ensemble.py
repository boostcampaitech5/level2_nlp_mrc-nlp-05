import collections
import argparse
import json
import pandas as pd
from datasets import load_from_disk

def probs_voting_ensemble(weights, path, number, test_df):
    """최고 probs 하나만을 고려하여 soft emsemble을 해주는 함수

    Args:
        weights (list): 각 predictions 별 가중치
        path (str): prediction이 저장되어 있는 폴더 경로
        number (int): ensemble 파일 개수
        test_df (pd.DataFrame): test 데이터 DataFrame
    """    
    
    test_ids = test_df['id'].tolist()
    nbest_prediction = collections.OrderedDict()
    prediction = collections.OrderedDict()
    weights = [weights[i] / sum(weights) for i in range(len(weights))]
    
    nbest_hubo = []
    best_hubo = []
    
    for i in range(number):
        nbest_path = f'{path}/nbest_predictions_{i}.json'
        best_path = f'{path}/predictions_{i}.json'
        
        with open(nbest_path, 'r') as json_file:
            json_data = json.load(json_file)
            nbest_hubo.append(json_data)
        with open(best_path, 'r') as json_file:
            json_data = json.load(json_file)
            best_hubo.append(json_data)
        

    for i in range(len(test_ids)):
        id = test_ids[i]
        max_doc_num = None
        max_probs = 0
        
        for j in range(number):
            pred = nbest_hubo[j][id][0]
            score = (pred["probability"]) * weights[j]
            
            if max_probs <= score:
                max_doc_num = j
                max_probs = score
                
        nbest_prediction[id] = nbest_hubo[max_doc_num][id]
        prediction[id] = best_hubo[max_doc_num][id]
        
    nbest_file = f'{path}/nbest_predictions.json'
    best_file = f'{path}/predictions.json'
    
    with open(nbest_file, "w", encoding="utf-8") as writer:
        writer.write(
            json.dumps(nbest_prediction, indent=4, ensure_ascii=False) + "\n"
        )
    with open(best_file, "w", encoding="utf-8") as writer:
        writer.write(
            json.dumps(prediction, indent=4, ensure_ascii=False) + "\n"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument(
        "--scores_list", nargs='+', type=float, help="list of float"
    )
    parser.add_argument(
        "--folder_path", default=f"/opt/ml/ensemble", type=str, help="folder path"
    )
    parser.add_argument(
        "--file_number", type=int, help="ensemble file number"
    )
    
    test_dataset = load_from_disk("/opt/ml/input/data/test_dataset")
    test_df = pd.DataFrame(test_dataset['validation'])
    
    args = parser.parse_args()
    
    probs_voting_ensemble(args.scores_list, args.folder_path, args.file_number, test_df)
    