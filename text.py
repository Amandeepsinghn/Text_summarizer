import validators
import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader


## streamlit App 
st.set_page_config(page_title="langchain: summarize text from a website",page_icon="A")
st.title(" langchain:Summarize Text from yt or website")
st.subheader("summarize")



## Get the Groq api key and url 

with st.sidebar:
    groq_api_key=st.text_input("groq api key",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

llm=ChatGroq(model="Gemma:2b",groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("summarize the content from yt or website"):
    ## Validate all the input
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("please provide the information to get started")

    elif not validators.url(generic_url):
        st.error("please enter a valid url. It can may be a yt video utl or website url")

    else:
        try:
            with st.spinner("wating....."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)

                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                    })
                
                docs=loader.load()


                ## Chain for summarization 
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)

                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.Exception(f"Exception{e}")
            