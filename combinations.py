datasets = ['polemo', 'yelp', 'imdb', 'amazon']
datasets = ['polemo']
datasets = ['cawi2-emo']
models = ['bert-base-cased', #'bert-base-uncased',
          'bert-large-cased', #'bert-large-uncased',
          'xlnet-base-cased', #'xlnet-base-uncased',
          'xlnet-large-cased', #'xlnet-large-uncased'
          'allegro/herbert-base-cased', 'allegro/herbert-large-cased'
          ]
polemo_variants = ['hotels', 'medicine', 'products', 'reviews']
target_gpu = '1'

for dataset in datasets:
    for model in models:
        print('CUDA_VISIBLE_DEVICES={} python main.py --transformer "{}" --dataset {}'.format(target_gpu, model, dataset))
    # for variant in polemo_variants:
    #     for model in models:
    #         print('CUDA_VISIBLE_DEVICES={} python main.py --transformer "{}" --dataset {} --polemo "{}_{}_{}"'.format(target_gpu, model, dataset, variant, variant, variant))
