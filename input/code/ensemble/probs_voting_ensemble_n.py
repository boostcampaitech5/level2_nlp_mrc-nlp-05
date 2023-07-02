import collections
import argparse
import json
import pandas as pd
from datasets import load_from_disk

def probs_voting_ensemble_n(weights, path, number, nbest, test_df):
    """ nbest의 probs를 고려하여 soft emsemble을 해주는 함수

    Args:
        weights (list): 각 nbest_predictions 별 가중치
        path (str): nbest_prediction이 저장되어 있는 폴더 경로
        number (int): ensemble 파일 개수
        nbest (int): 몇 개의 nbest까지 ensemble에 고려할 것인지의 개수
        test_df (pd.DataFrame): test 데이터 DataFrame
    """    
    
    test_ids = test_df['id'].tolist()
    prediction = collections.OrderedDict()
    weights = [weights[i] / sum(weights) for i in range(len(weights))]
    
    nbest_hubo = []
    
    for i in range(number):
        nbest_path = f'{path}/nbest_predictions_{i}.json'
        
        with open(nbest_path, 'r') as json_file:
            json_data = json.load(json_file)
            nbest_hubo.append(json_data)
    
    for id in test_ids:
        hubo = collections.defaultdict(float)
        
        for i in range(number):
            preds = nbest_hubo[i][id][:nbest]
            for pred in preds:
                hubo[pred["text"]] += pred["probability"] * weights[i]
                
        max_text = max(hubo, key=hubo.get)
        prediction[id] = max_text
                
        
    best_file = f'{path}/predictions.json'
    
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
    parser.add_argument(
        "--nbest", default=3, type=int, help="nbest to include"
    )
    
    test_dataset = load_from_disk("/opt/ml/input/data/test_dataset")
    test_df = pd.DataFrame(test_dataset['validation'])
    
    args = parser.parse_args()
    
    probs_voting_ensemble_n(args.scores_list, args.folder_path, args.file_number, args.nbest, test_df)
    