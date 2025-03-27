# NLP Analysis on Project Gutenberg Books   

## Project Overview  
This project applies **Natural Language Processing (NLP)** techniques to analyze books from **Project Gutenberg** using **Python**, **NLTK**, and **WordCloud**.  

With a single execution, the Python script will:  
- Load a book’s text from a **Project Gutenberg URL**  
- Perform **tokenization, stopword removal, and lemmatization**  
- Generate **word frequency and bigram distributions**  
- Calculate the **Lexical Diversity Score**  
- Visualize the text using a **Word Cloud**  

## Example Book Used  
This project processes the book:  
**[The Story of the Alphabet (Ebook #64317)](https://www.gutenberg.org/ebooks/64317)**  

Direct text file URL: https://www.gutenberg.org/ebooks/64317

## Example Output  
### Word Frequency Table  
| Word  | Frequency |  
|--------|-----------|  
| alphabet | 120 |  
| letter | 95 |  
| writing | 75 |  

### Word Cloud  
![output image](https://github.com/user-attachments/assets/ec844654-a95c-47d5-9126-f26e7ebc9d10)


## Project Structure  
```
NLP-Gutenberg-Analysis/
│── NLP_Gutenberg_Analysis.ipynb   # Google Colab notebook (Python code)
│── README.md                      # Project Documentation
│── wordcloud.png                  # Output Image (Word Cloud)
│── requirements.txt                # Dependencies (if needed)
```

## Setup & Execution
# Option 1: Run in Google Colab

Open the Google Colab Notebook

Run all cells

Enter a Project Gutenberg URL when prompted

## Technologies Used

Python (NLTK, Matplotlib, Seaborn, WordCloud, Pandas)

Google Colab / Jupyter Notebook

Requests (Fetching book text)

## Future Enhancements

Named Entity Recognition (NER) for character analysis

Sentiment analysis for book reviews

Interactive dashboard using Streamlit



