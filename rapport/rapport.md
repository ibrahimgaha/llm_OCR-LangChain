# Projet 4 — OCR avec LangChain et NVIDIA AI

**Étudiant** Ibrahim Gaha

## Objectif

La reconnaissance optique de caractères (OCR) transforme le texte visible
dans une image en texte sélectionnable et modifiable. Ce projet permet de
transcrire une image PNG ou JPEG en conservant autant que possible la
langue, l’orthographe et les retours à la ligne d’origine.

## Technologies

- **Streamlit** gère le chargement de l’image, l’aperçu et le téléchargement du résultat.
- **Pillow** vérifie le format et le contenu de l’image.
- **LangChain** construit le message multimodal avec `ChatPromptTemplate`.
- **ChatNVIDIA** envoie le message à `mistralai/mistral-large-4-1t-a52b`.
- **python-dotenv** charge la clé API depuis le fichier local `.env`.

## Architecture

`app.py` reçoit le fichier et affiche son aperçu après validation.
`ocr.py` encode l’image en base64 et l’ajoute au message sous la forme
d’une URL de données. Le message contient aussi la consigne de transcription.
L’interface affiche le texte retourné et permet de le copier ou de le
télécharger au format TXT.

## Résultats

Les tests hors ligne vérifient les formats PNG et JPEG, le rejet des fichiers
invalides, la structure de la requête multimodale, les réponses vides ou
tronquées et les erreurs réseau. Le test Streamlit vérifie le parcours de
chargement et d’extraction avec une transcription simulée. La qualité de
la transcription du modèle hébergé reste à vérifier avec un appel réel.
