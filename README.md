# SpaceBot - Ask me anything about space.

## Beskrivning:

En RAG-chattbot som svarar på frågor om solsystemet. All fakta och information som chattbotten använder kommer från en PDF-fil från NASA som heter "Our Solarsystem". Chattbotten ska kunna svara på allt från enkla frågor till mer detaljerade frågor om vårt solsystem. 

## Hämta repot:

- git clone <repo-url>

## Virtual Environment:

- uv init 

- uv add jupyter


## Funktioner:

1. Inläsning av PDF-filen.
2. Chunk-indelning.
3. Funktion som skapar embeddings.
4. Funktion som kör en semantisk sökning.
5. Generera svar med hjälp av en prompt.
6. Evaluering av svar från chattbotten.

## Filer: 

chatbot.ipynb - Hela RAG-flödet byggs i denna fil.

rag-bot.pk - Sparade chunks och embeddings från chatbot.ipynb som hämtas sedan i app.py.

app.py - Applikationen där chattbotten är implementerad.

Starta applikationen:
 - uv run streamlit run app.py


