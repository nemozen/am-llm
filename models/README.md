# Train the translation model with parallel text
```
cd ../..
git clone https://github.com/MarsPanther/Amharic-English-Machine-Translation-Corpus
cd am-llm/models
./am2en.py --train 32
```

# Predict, i.e. translate from amharic on stdin to english on stdout
```
echo "ሰላም ዓለም" | ./am2en.py --predict
```