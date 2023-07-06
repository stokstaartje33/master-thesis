import shutil

from senda import Model, compute_metrics
from sklearn.metrics import classification_report
from transformers import EarlyStoppingCallback
from transformers.integrations import WandbCallback
import wandb
import argparse
from datasets import DanishTweetsDataset, AmazonDataset, YelpDataset, IMDbDataset, PolEmoDataset, ClarinEmoDataset, \
    ClarinEmoEmoDataset, CawiTwoDataset, CawiTwoAroDataset, CawiTwoEmoDataset

# Create the parser
parser = argparse.ArgumentParser()

# Add an argument
parser.add_argument('--transformer', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--repetitions', type=int, default=10)
parser.add_argument('--devmode', type=bool, default=False)
parser.add_argument('--polemo', type=str, default="all_all_all")

if __name__ == "__main__":
    # Parse the argument
    args = parser.parse_args()

    repetitions = args.repetitions
    transformer = args.transformer
    devmode = args.devmode
    dataset = None
    m = None

    if args.dataset == "danish-tweets":
        dataset = DanishTweetsDataset()
    elif args.dataset == "amazon" or args.dataset == "amazon-polarity":
        dataset = AmazonDataset()
    elif args.dataset == "yelp":
        dataset = YelpDataset()
    elif args.dataset == "imdb":
        dataset = IMDbDataset()
    elif args.dataset == "polemo":
        dataset = PolEmoDataset()
        trainset, evalset, testset = args.polemo.split("_")
        dataset.read(trainset, "train")
        dataset.read(evalset, "dev")
        dataset.read(testset, "test")
    elif args.dataset == "clarinemo":
        dataset = ClarinEmoDataset()
    elif args.dataset == "clarinemo-emo":
        dataset = ClarinEmoEmoDataset()
    elif args.dataset == "cawi2":
        dataset = CawiTwoDataset()
    elif args.dataset == "cawi2-aro":
        dataset = CawiTwoAroDataset()
    elif args.dataset == "cawi2-emo":
        dataset = CawiTwoEmoDataset()
    else:
        print('Dataset undetected or does not exist')
        exit()

    dataset.split()
    # print(dataset.df_train.shape, dataset.df_eval.shape, dataset.df_test.shape)
    # exit()
    if devmode:
        dataset.limit()

    for i in range(repetitions):
        test_name = transformer + '_' + dataset.name() + '_' + str(i)
        test_tags = [transformer, dataset.name(), str(i)]
        if dataset.name() == "polemo":
            test_tags.append(args.polemo)
        wandb.init(project="final-run",
                   entity="ai-dream-team",
                   name=test_name,
                   tags=test_tags,
                   group=dataset.name())

        m = Model(train_dataset=dataset.df_train,
                  eval_dataset=dataset.df_eval,
                  transformer=transformer,
                  labels=dataset.label(),
                  tokenize_args={'padding': True, 'truncation': True, 'max_length': 512},
                  training_args={"output_dir": './{}'.format(test_name),
                                 "num_train_epochs": 4,
                                 "per_device_train_batch_size": 4,
                                 "evaluation_strategy": "steps",
                                 "eval_steps": 100,
                                 "save_steps": 100,
                                 "logging_steps": 100,
                                 "learning_rate": 2e-05,
                                 "weight_decay": 0.01,
                                 "per_device_eval_batch_size": 32,
                                 "warmup_steps": 100,
                                 "seed": 42,
                                 "load_best_model_at_end": True,
                                 "metric_for_best_model": "f1"
                                 },
                  trainer_args={'compute_metrics': compute_metrics,
                                'callbacks': [EarlyStoppingCallback(early_stopping_patience=4),
                                              WandbCallback],
                                }
                  )

        # initialize Trainer
        m.init()
        # run training
        m.train()

        # run final evaluation on df_test
        m.evaluate(dataset.df_test)

        # clean after the training
        if test_name is not None and len(test_name) > 0:
            shutil.rmtree('./{}'.format(test_name))

        wandb.finish()

    # labels = m.predict(dataset.df_test, return_labels=True)
    # print(classification_report(dataset.df_test['label'].tolist(), labels))
