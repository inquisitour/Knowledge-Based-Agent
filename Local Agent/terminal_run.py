
from agent import ResponseAgent
from data_processing import DBops
import pandas as pd

def main():
    

    # Ensure the database is setup before handling any events
    db_ops = DBops()
    db_ops.setup_database()

    # Two file uploaders for Excel and CSV files
    #uploaded_excel = st.file_uploader("Upload your Excel data file", type=['xlsx'], key='excel')
    uploaded_csv = "C:/Users/Rudra/main/code/gravitas/Knowledge-Based-Agent/Local Agent/categorized_qa_pairs.csv"

    #if uploaded_excel is not None and uploaded_csv is not None:
        #data_excel = pd.read_excel(uploaded_excel)
    if uploaded_csv is not None:
        data_csv = pd.read_csv(uploaded_csv)
        db_ops.process_local_file(data_csv)

    agent = ResponseAgent()
    print("Agent initialised!")

    # User query section
    while True:
        user_question = input("-----------\nenter a question\n-----------\n")
        if user_question == "exit":
            break
        if user_question:
            
            response = agent.answer_question(user_question)
            print(response)

    # Adding a Lottie animation for loading

if __name__ == "__main__":
    main()
