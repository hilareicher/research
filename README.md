<h2>Instructions</h2>
1. Set environment vars: EMR_INPUT_DIR and SAFE_HARBOR_ENDPOINT, e.g.,

```
   EMR_INPUT_DIR=/home/var/EMRs
   
   SAFE_HARBOR_ENDPOINT=http://127.0.0.1:8000/query
```

2. Run *extract_entities.py*. A successful run will generate the files *response.json* and *entities.csv*
3. Manually anonymized entities by editing the file *entities.csv*.
   The last column is the replacement string. If no value is specified, then the text will keep the original value. 
4. Run *replace_entities.py*. A successful run will generate *response_with_replacements.json* 
