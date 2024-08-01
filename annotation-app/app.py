import streamlit as st
import json
from collections import Counter


def load_json(file):
    return json.load(file)


def display_json_item(data, index):
    item = data[index]
    for key, value in item.items():
        st.write(f"**{key}:** {value}")
        if key == 'synthesis':
            st.write(f"**LEN Synthesis:** {len(value.split())}")
        if key == "reward":
            # Use st.text_input to get the updated value
            updated_value = st.text_input(f"{key}:", value, key=f"{index}_{key}")
            # Update the item in the data list
            item[key] = int(updated_value)  # Assuming reward is an integer


    # Calculate reward statistics
    reward_values = [item['reward'] for item in data]
    reward_counts = Counter(reward_values)

    # Display reward statistics
    st.write("**Reward Statistics:**")
    st.write(f"Value -1: {reward_counts.get(-1, 0)} times")
    st.write(f"Value 0: {reward_counts.get(0, 0)} times")
    st.write(f"Value 1: {reward_counts.get(1, 0)} times")


def save_json(data, file):
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    st.title("JSON Viewer and Editor")

    uploaded_file = st.file_uploader("Upload JSON file", type="json")

    if uploaded_file is not None:
        if 'data' not in st.session_state:
            st.session_state.data = load_json(uploaded_file)

        data = st.session_state.data
        total_items = len(data)
        st.write(f"Total items: {total_items}")

        if 'index' not in st.session_state:
            st.session_state.index = 0

        if st.button("Previous"):
            if st.session_state.index > 0:
                st.session_state.index -= 1

        st.write(f"Current Item: {st.session_state.index + 1} / {total_items}")

        if st.button("Next"):
            if st.session_state.index < total_items - 1:
                st.session_state.index += 1

        display_json_item(data, st.session_state.index)

        if st.button("Save Changes"):
            save_json(data, uploaded_file.name)
            st.success("Changes saved to updated_data.json")


if __name__ == "__main__":
    main()
