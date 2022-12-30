import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import evaluation
from torch.utils.data import DataLoader

df = pd.read_csv('data/fold_split.csv')

FOLD = 0

train_df = df[df.fold != FOLD]
val_df = df[df.fold == FOLD]

def gen_input_examples(df):
    examples = []
    for i, row in df.iterrows():
        ex = InputExample(texts=[row['topic_title'], row['content_title'] ], label=float(row['match']))
        examples.append(ex)
    return examples


#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

#Define your train examples. You need more than just two examples...
train_examples = gen_input_examples(train_df)

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1)
train_loss = losses.CosineSimilarityLoss(model)

evaluator = evaluation.EmbeddingSimilarityEvaluator(val_df['topic_title'], val_df['content_title'], val_df['match'])

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=evaluator, evaluation_steps=500, epochs=1, warmup_steps=100)