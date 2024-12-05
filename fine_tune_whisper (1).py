# -*- coding: utf-8 -*-
from datasets import Audio
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from datasets import load_dataset, DatasetDict
import datasets
import argparse
from transformers import pipeline
from datetime import datetime
#import gradio as gr
from transformers import WhisperFeatureExtractor
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from util import DataCollatorSpeechSeq2SeqWithPadding
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
import transformers

from huggingface_hub import login
hub_key = ""
login(hub_key)
#transformers.logging.set_verbosity_info()


def main():
    common_voice = DatasetDict()
    # argParser = argparse.ArgumentParser()
    # argParser.add_argument("-n","--folder_name",help="Enter folder name")
    # args = argParser.parse_args()
    common_voice = load_dataset("audiofolder",data_dir="")
    pretrained_model_path = "openai/whisper-tiny.en"
    output_model_path = "openai/whisper-tiny-en-blink"
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))



    feature_extractor = WhisperFeatureExtractor.from_pretrained(pretrained_model_path)
    tokenizer = WhisperTokenizer.from_pretrained(pretrained_model_path, task="transcribe")
    processor = WhisperProcessor.from_pretrained(pretrained_model_path, task="transcribe")
    print(common_voice)
    input_str = common_voice["train"][0]["transcription"]
    labels = tokenizer(input_str).input_ids
    decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
    decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)





    """Let's initialise the data collator we've just defined:"""

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    import evaluate
    metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_path)

    model.config.forced_decoder_ids = None
    # model.config.suppress_tokens = []

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_model_path,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=1000,
        gradient_checkpointing=True,
        #fp16=True,
        fp16=False,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=50,
        eval_steps=50,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
        hub_token=hub_key,
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice['train'],
        eval_dataset=common_voice['train'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        #hub_token='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',
    )

    processor.save_pretrained(training_args.output_dir, safe_serialization=False)
    start_time = datetime.now()
    print("Training started AT: {}".format(start_time))
    trainer.train()
    end_time = datetime.now()
    print("Training finished AT: {}".format(end_time))
    print("Trained in: {} secs".format((end_time - start_time).total_seconds()))

    kwargs = {
        "dataset_tags": "mozilla-foundation/common_voice_11_0",
        "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
        "dataset_args": "config: hi, split: test",
        "language": "hi",
        "model_name": "Whisper Small Hi - Sanchit Gandhi",  # a 'pretty' name for our model
        "finetuned_from": "nurzhanit/whisper-enhanced-ml",
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }

    trainer.push_to_hub(**kwargs)




    '''
    iface = gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(source="microphone", type="filepath"),
        outputs="text",
        title="Whisper Small Hindi",
        description="Realtime demo for Hindi speech recognition using a fine-tuned Whisper small model.",
    )

    iface.launch()
    '''

if __name__ == "__main__":
    main()

