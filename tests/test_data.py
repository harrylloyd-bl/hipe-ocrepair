import json
import ocrepair.postcorrect as pc
import pytest
from huggingface_hub import InferenceClient


def test_create_client_with_token():
    client = pc.get_client()
    assert type(client.token) == str


@pytest.mark.paid()
def test_client_inference():
    client = pc.get_client()
    model = "google/gemma-3-27b-it"
    messages = [
        {"role": "system", "content": "This question is part of a test suite"},
        {"role": "user", "content": "Please respond with the word ocrepair if functioning"},
    ]

    result = client.chat.completions.create(
                model=model,
                messages=messages
            )
    
    message = result.choices[0].message.content
    assert message == "ocrepair\n"


@pytest.mark.paid()
def test_chat_completion():
    client = pc.get_client()
    model = "google/gemma-3-27b-it"
    messages = [
        {"role": "system", "content": "This question is part of a test suite"},
        {"role": "user", "content": "Please respond with the word ocrepair if functioning"},
    ]

    message, finish_reason = pc.chat_completion(
                client=client,
                model=model,
                messages=messages
            )
    
    assert message == "ocrepair\n"
    assert finish_reason == "stop"


def test_resolve_model():
    model = "qwen3.5-397b-a17b"
    resolved = pc.resolve_model(model)
    assert resolved == "Qwen/Qwen3.5-397B-A17B"

    not_found_model = "TestModel/TestModel1-23B"
    not_found_resolved = pc.resolve_model(not_found_model)
    assert not_found_resolved == "TestModel/TestModel1-23B"


def test__strip_fences():
    # single line strings as pytest interprets the indents as tabs
    s = "```json\nlorem ipsum\n```"
    stripped = pc._strip_fences(s)
    assert stripped == "lorem ipsum"

    multiline_s = "```json\nlorem ipsum\ndolor sit amet\n```"
    multiline_stripped = pc._strip_fences(multiline_s)
    assert multiline_stripped == "lorem ipsum\ndolor sit amet"

    multiline_no_lb_s = "```jsonlorem ipsum\ndolor sit amet```"
    multiline_no_lb_stripped = pc._strip_fences(multiline_s)
    assert multiline_stripped == "lorem ipsum\ndolor sit amet"


def test__try_parse_json(capsys):
    correct_json = '```json\n{"key":"value"}\n```'
    incorrect_json = "{'a'}"
    jsonl_format = '```json\n{"key0": "value0"}\n{"key1": "value1"}\n```'
    
    parse_correct = pc._try_parse_json(correct_json)
    assert parse_correct == {"key": "value"}

    with pytest.raises(json.JSONDecodeError):
        pc._try_parse_json(incorrect_json)

    parse_jsonl = pc._try_parse_json(jsonl_format)
    captured = capsys.readouterr()
    assert parse_jsonl == [{"key0": "value0"}, {"key1": "value1"}]
    # stderr should have the error from failing to parse incorrect_json as well as from the initial failure on parsing jsonl_format
    expected_err = "  (initial json.loads failed: Expecting property name enclosed in double quotes: line 1 column 2 (char 1))\n  (initial json.loads failed: Extra data: line 2 column 1 (char 19))\n"
    assert captured.err == expected_err
    

def test__split_json_from_rest():
    assert pc._split_json_from_rest("") == ("", "")
    
    no_json = "lorem ipsum"
    assert pc._split_json_from_rest(no_json) == ("lorem ipsum", "")

    bad_json = '{"key0": bad_value}'
    assert pc._split_json_from_rest(bad_json) == ('{"key0": bad_value}', "")

    splittable = '```jsonloremipsum{"key0": "value1"}```asdfasdf  '
    split_json, end = pc._split_json_from_rest(splittable)
    assert split_json == '{"key0": "value1"}'
    assert end == "```asdfasdf"


def test_parse_response():
    corrections = """
    ```json
    {"key0": "value0"}
    ```
    CORRECTIONS_CSV
    col0, col1, col2
    val0, val1, val2
    """

    parse_corrections = pc.parse_response(corrections)
    assert parse_corrections[0] == '{"key0": "value0"}'
    assert parse_corrections[1] == "col0, col1, col2\n    val0, val1, val2"

    multi_block = """
    ```json
    {"key0": "value0"}
    ```

    ```
    col0, col1, col2
    val0, val1, val2
    ```
    """

    parse_multi = pc.parse_response(multi_block)
    assert parse_multi[0] == '{"key0": "value0"}'
    assert parse_multi[1] == "col0, col1, col2\n    val0, val1, val2"

    single_block = """
    ```json
    {"key0": "value0"}
    
    col0, col1, col2
    val0, val1, val2
    ```
    """

    parse_single = pc.parse_response(single_block)
    assert parse_single[0] == '{"key0": "value0"}'
    assert parse_single[1] == "col0, col1, col2\n    val0, val1, val2"

    no_block = """
    {"key0": "value0"}
    
    col0, col1, col2
    val0, val1, val2
    """

    parse_no_block = pc.parse_response(no_block)
    assert parse_no_block[0] == '{"key0": "value0"}'
    assert parse_no_block[1] == "col0, col1, col2\n    val0, val1, val2"