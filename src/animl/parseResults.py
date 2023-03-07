import pandas as pd


def parseMD(results):
    if len(results) > 0:
        df = pd.DataFrame()
        for dictionary in results:
            detections = dictionary['detections']
            for detection in detections:
                bbox = detection['bbox']
                data = {'file': dictionary['file'], 'max_detection_conf': dictionary['max_detection_conf'],
                        'category': detection['category'], 'conf': detection['conf'], 'bbox1': bbox[0],
                        'bbox2': bbox[1],
                        'bbox3': bbox[2], 'bbox4': bbox[3]}
                df = df.append(data, ignore_index=True)
        return df


# def parseMDjson

def applyPredictions(animaldf, otherdf, predictions, classes):
    # Format Classification results
    table = pd.read_table(classes, sep=" ", index_col=0)
    
    animaldf['class'] = [table['x'].values[int(np.argmax(x))] for x in predictions]
    
    if(otherdf != None):
      maxDataframe = animaldf.append(otherdf, ignore_index=True)

    else:
      maxDataframe = animaldf
    # Read Classification Txt file

    return maxDataframe
