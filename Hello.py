import streamlit as st
from haystack.pipelines import Pipeline
from haystack.nodes import EmbeddingRetriever
import pandas as pd
import joblib

# Function to load the retriever and document store
def load_resources():
    if "document_store" not in st.session_state or "retriever" not in st.session_state:
        st.session_state.document_store = joblib.load("doc_str_joblib.joblib")
        st.session_state.retriever = EmbeddingRetriever(
            document_store=st.session_state.document_store, 
            embedding_model="BAAI/bge-large-en"
        )
        st.session_state.pipeline = Pipeline()
        st.session_state.pipeline.add_node(component=st.session_state.retriever, name="retriever", inputs=["Query"])


st.session_state.df = pd.read_csv("processed_text.csv")



# Streamlit interface

def update_output(query, slider_value):
    result = st.session_state.pipeline.run(
        query=query,
        params={
            "retriever": {"top_k": slider_value},
        },
    )  # "Reader": {"top_k": 1}})
    catch_list = []
    for pt_link_index in range(0,len(result['documents'])):
        pt_link = result['documents'][pt_link_index].to_dict()['meta']['source']
        text_for_output = st.session_state.df[st.session_state.df['patient_fix'] == pt_link]['raw_text'].iloc[0]
        # text_for_output = text_for_output.replace('\n',' ')
        catch_list.append(text_for_output)

    warmups = [result.split('Warmup: ')[1].split('Cooldown: ')[0].replace(': ','') for result in catch_list]
    cooldowns = [result.split('Cooldown: ')[1].replace(': ','') for result in catch_list]
    exercises = [result.split('Exercises')[1].replace(': ','') for result in catch_list]

    zipped_list = list(zip(warmups, cooldowns, exercises))

    # for i in range(0,slider_value):
    #     print('query for search',query,'\n')
    #     print('returned results:')
    #     print('warmups:',zipped_list[i][0],'\n')    
    #     print('cooldowns:', zipped_list[i][1],'\n')
    #     print('exercises:', zipped_list[i][2],'\n')
    out_df = pd.DataFrame(zipped_list, 
                columns=['warmups', 'cooldowns' , 'exercises'])

    # # The output for each section is simply the slider value
    # # warmup = ['warmup1','warmup2']
    # # cooldown = ['cooldown1', 'cooldown2']
    # # exercises = ['exercise1', 'exercise2']
    # # return warmup, cooldown, exercises
    # result1 = ["result1", "result1", "result1", "result1"]
    # result2 = ["result2", "result2", "result2", "result2"]
    # df = pd.DataFrame(
    #     [result1, result2], columns=["rank", "warmup", "cooldown", "exercises"]
    # )

    return out_df

# Streamlit main body
if __name__ == "__main__":
    st.title("UMICH Similarity App")
    
    load_resources()
    
    
    with st.form("my_form"):
        query = st.text_input("Screening Findings")
        slider_value = st.slider("Results to match", 1, 5, 1)

        # Every form must have a submit button.
        submitted = st.form_submit_button("Search")
        if submitted:
            results = update_output(query, slider_value)
            st.dataframe(results)


# No need for a separate launch command
