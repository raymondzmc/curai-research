import json, pdb


# import json

# def normalize_string(s):
#     return ''.join([c for c in s.split('[SEP]')[0].lower() if c.isalpha()])

# if __name__=='__main__':
#     annotations_path='annotations.json'
#     cursa_path='resources/CuRSA/CuRSA-FIXED-v0-processed-all.json'
#     with open(annotations_path,'r') as fin:
#         annotations=json.load(fin)
#     with open(cursa_path,'r') as fin:
#         cursa=json.load(fin)

#     annotation_hash_is_multi = dict()
#     for annotation in annotations:
#         if annotation['classes']:
#             h = (normalize_string(annotation['annotations'][0][0]), annotation['classes'][0])
#         annotation_hash_is_multi[h]=len(annotation['annotations'][0][1]['entities'])

#     for datum in cursa['data']:
#         datum['is_multi']=dict()
#         for entity in datum['entities']:
#             h = (normalize_string(datum['text']), entity)
#             datum['is_multi'][entity]=annotation_hash_is_multi.get(h,0)
#     with open('cursa_with_multi_annot.json','w') as fout:
#         json.dump(cursa,fout,indent=2)


def normalize_string(s):
    return ''.join([c for c in s.split('[SEP]')[0].lower() if c.isalpha()])

if __name__=='__main__':
    annotations_path='annotations.json'
    cursa_path='resources/CuRSA/CuRSA-FIXED-v0-processed-all.json'
    with open(annotations_path,'r') as fin:
        annotations=json.load(fin)
    with open(cursa_path,'r') as fin:
        cursa=json.load(fin)


    annotation_hash = dict()
    annotation_hash_is_multi = dict()
    for annotation in annotations:
        if annotation['classes']:

            h = (normalize_string(annotation['annotations'][0][0]), annotation['classes'][0])

        annotation_hash[h]=annotation['annotations']
        annotation_hash_is_multi[h]=len(annotation['annotations'][0][1]['entities'])

    for datum in cursa['data']:
        datum['annotations']=dict()
        for entity in datum['entities']:
            h = (normalize_string(datum['text']), entity)
            datum['annotations'][entity]=annotation_hash.get(h,0)
            datum['is_multi'][entity]=annotation_hash_is_multi.get(h,0)

    with open('cursa_with_span_annot.json','w') as fout:
        json.dump(cursa,fout,indent=2)