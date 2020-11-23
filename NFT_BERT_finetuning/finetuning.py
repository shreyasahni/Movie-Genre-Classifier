from finetuning_functions import *

# Set Data and Label columns
DATA_COLUMN = "overview"
LABEL_COLUMN = "genre"
LABEL_MAP = ["a", "b", "c"] #RECHECK

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1" #RECHECK

# Set output directory to save model checkpoints
OUTPUT_DIR = "C:\\Users\\Anushka Chincholkar.LAPTOP-PVP8F6JH\\Desktop\\College\\Python Projects\\NFT_BERT_finetuning\\model" #RECHECK

# We'll set sequences length
MAX_SEQ_LENGTH = 175

# Compute train and warmup steps from batch size
BATCH_SIZE = 3 #RECHECK
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 3.0 #RECHECK
# Warmup is a period of time where the learning rate is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

# Set filepaths to train and test dataset
meta_train = pd.read_csv("trainer.csv") #RECHECK
meta_test = pd.read_csv("tester.csv") #RECHECK

# Creating Training DataFrame
print("***** Loading Train Dataframe *****")
train = pd.DataFrame({DATA_COLUMN: meta_train['overview'], LABEL_COLUMN: meta_train['genre']}) #RECHECK
print(train.columns)
print(train.head)

# Creating Test DataFrame
print("***** Loading Test Dataframe *****")
test = pd.DataFrame({DATA_COLUMN: meta_test['overview'], LABEL_COLUMN: meta_test['genre']}) #RECHECK
print(test.columns)
print(test.head)

# Convert training and test examples to the BERT Input Examples
train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this application
                                                         text_a = x[DATA_COLUMN], 
                                                         text_b = None, 
                                                         labels = prepare_input(x[LABEL_COLUMN])), axis = 1)

test_InputExamples = test.apply(lambda x: InputExample(guid=None, 
                                                       text_a = x[DATA_COLUMN], 
                                                       text_b = None, 
                                                       labels = prepare_input(x[LABEL_COLUMN])), axis = 1)
print(test_InputExamples[1].labels)

print("***** Creating Tokenizer *****")
print("This will take a few minutes the first time it is run")
tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_HUB)

# print(tokenizer.tokenize("This here's an example of using the BERT tokenizer"))

# Convert our train and test features to InputFeatures that BERT understands.
print("***** Tokenizing and converting training to InputFeatures *****")
train_features = convert_examples_to_features(train_InputExamples, MAX_SEQ_LENGTH, LABEL_MAP, tokenizer)
print(train_features[0].input_ids)

print("***** Tokenizing and converting testing to InputFeatures *****")
test_features = convert_examples_to_features(test_InputExamples, MAX_SEQ_LENGTH, LABEL_MAP, tokenizer)
print(test_features[1].label_ids)

# Compute # train and warmup steps from batch size
num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Specify output directory and number of checkpoint steps to save
run_config = tf.estimator.tpu.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)

#Create madel function to pass to estimator
model_fn = model_fn_builder(
  num_labels=len(LABEL_MAP),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps,
  use_tpu=False,
  bert_hub_module_handle=BERT_MODEL_HUB)

#Create the tensorflow estimator that will train and evaluate the model
estimator = tf.estimator.tpu.TPUEstimator(
  model_fn=model_fn,
  config=run_config,
  train_batch_size=BATCH_SIZE,
  eval_batch_size=BATCH_SIZE,
  params={},
  use_tpu=False,
  eval_on_tpu=False,
  export_to_tpu=False,
  export_to_cpu=True)

# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    num_labels=len(LABEL_MAP),
    is_training=True,
    drop_remainder=False)

print(f'Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time ", datetime.now() - current_time)

test_input_fn = input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    num_labels=len(LABEL_MAP),
    is_training=False,
    drop_remainder=False)

eval_metrics = estimator.evaluate(input_fn=test_input_fn, steps=None)
print("Test Set results: ")
print(eval_metrics)
