import csv
import os
from typing import List
import yaml
import asyncio
import json
from api_provider import generate_response
import Levenshtein
from tqdm.asyncio import tqdm

# Load configuration from the YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def calculate_cer(ocr_text, gt_text):
    """
    Calculate the Character Error Rate (CER) between an OCR output and ground truth (GT).

    Args:
        ocr_text (str): The OCR output text.
        gt_text (str): The ground truth text.

    Returns:
        float: The Character Error Rate (CER).
    """
    edit_distance = Levenshtein.distance(ocr_text, gt_text)
    total_gt_chars = len(gt_text)

    if total_gt_chars == 0:
        return float("inf") if len(ocr_text) > 0 else 0.0
    cer = edit_distance / total_gt_chars
    return cer


def dataloader():
    dataset_folder = "dataset"
    csv_file_path = os.path.join(dataset_folder, "train.csv")

    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(
            f"The file '{csv_file_path}' does not exist in the '{dataset_folder}' folder."
        )

    parsed_data = []

    # Read and parse the CSV file
    with open(csv_file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed_data.append(row)
    finetune_data = []
    # Create finetune data with system and user prompts
    for d in parsed_data:
        finetune_data.append(
            {
                "instruction": config["prompt"]["system_prompt"],
                "input": config["prompt"]["user_prompt"] + d["OCR Text"],
                "output": d["Ground Truth"],
            }
        )
    with open("dataset/finetune_data.json", "w") as file:
        json.dump(finetune_data, file, indent=4)
    return parsed_data


lock = asyncio.Lock()


async def benchmark_model(model_name: str, data: List, provider: str):
    print(f"Benchmarking {model_name} with {provider}")
    system_prompt = config["prompt"]["system_prompt"]
    user_prompt = config["prompt"]["user_prompt"]
    tasks = []
    responses = []

    # Generate responses asynchronously for each data entry
    for d in tqdm(data, desc="Processing", total=len(data)):
        response = await generate_response(
            provider, model_name, system_prompt, user_prompt + d["OCR Text"]
        )
        responses.append(response)
    cer_list = []
    output_list = []
    cer_reduction_list = []
    ground_truth_list = [d["Ground Truth"] for d in data]

    # Calculate CER and CER reduction
    for i in range(len(responses)):
        cer = calculate_cer(responses[i], data[i]["Ground Truth"])
        original_cer = float(data[i]["CER"])
        if original_cer == 0:
            cer_reduction = 0.0
        else:
            cer_reduction = (original_cer - cer) / original_cer * 100
        cer_list.append(cer)
        output_list.append(responses[i])
        cer_reduction_list.append(cer_reduction)

    # Calculate average CER and CER reduction
    average_cer = sum(cer_list) / len(cer_list) if cer_list else 0.0
    average_cer_reduction = (
        sum(cer_reduction_list) / len(cer_reduction_list) if cer_reduction_list else 0.0
    )

    result_data = {
        "model_name": model_name,
        "provider": provider,
        "cer_list": cer_list,
        "ground_truth_list": ground_truth_list,
        "output_list": output_list,
        "average_cer": average_cer,
        "average_cer_reduction": average_cer_reduction,
        "cer_reduction_list": cer_reduction_list,
    }

    async with lock:
        try:
            with open("results.json", "r") as file:
                original_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            original_data = []
        original_data.append(result_data)
        with open("results.json", "w") as file:
            json.dump(original_data, file, indent=4)

    return (
        result_data["model_name"],
        result_data["provider"],
        result_data["average_cer"],
        result_data["average_cer_reduction"],
    )


async def benchmark(data: List):
    with open("results.json", "w") as file:
        json.dump([], file, indent=4)
    results = []
    for provider in config["models"]:
        for model in config["models"][provider]:
            result = await benchmark_model(model, data, provider)
            results.append(result)
    for result in results:
        print(
            f"Model: {result[0]}, Provider: {result[1]}, Average CER: {result[2]}, Average CER Reduction: {result[3]}"
        )
    summary = {"results": results}
    with open("summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=4)
    return results


if __name__ == "__main__":
    # Load and prepare the data
    data = dataloader()

    # Uncomment the following lines to run the benchmark
    # data = data[:100]  # Use a subset of data for testing
    # asyncio.run(benchmark(data))
