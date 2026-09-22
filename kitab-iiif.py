import os
import duckdb
import matplotlib.pyplot as plt

# 1. Parse raw text file list into structured tuples
file_data = []

with open("filelist.txt", "r") as f:
    text = f.read()

# Separate listings by directory markers
sections = text.split("./")
for section in sections:
    if not section.strip():
        continue
    lines = section.strip().splitlines()
    dir_header = lines[0].strip().rstrip(":")
    
    # Process files within the current directory section
    file_tokens = " ".join(lines[1:]).split()
    for token in file_tokens:
        clean_name = os.path.basename(token)
        if clean_name and not clean_name.startswith('.'):
            file_data.append((dir_header, clean_name))

# 2. Query and organize file metadata using DuckDB
con = duckdb.connect()
con.execute("CREATE TABLE raw_files(folder VARCHAR, filename VARCHAR)")
con.executemany("INSERT INTO raw_files VALUES (?, ?)", file_data)

query = """
SELECT 
    folder,
    filename,
    TRY_CAST(regexp_extract(filename, 'p([0-9]+)', 1) AS INT) AS page_num,
    TRY_CAST(regexp_extract(filename, 'v([0-9]+)', 1) AS INT) AS version_num,
    TRY_CAST(regexp_extract(filename, 'line([0-9]+)', 1) AS INT) AS line_num
FROM raw_files
ORDER BY folder, page_num NULLS LAST, version_num NULLS LAST, line_num NULLS LAST
"""

df_files = con.execute(query).df()
print("Organized Files Preview:")
print(df_files.head(10))

# 3. Aggregation & Visualization with Matplotlib
summary_query = """
SELECT 
    page_num, 
    COUNT(DISTINCT filename) AS line_count
FROM df_files
WHERE folder = 'lineimages'
GROUP BY page_num
ORDER BY page_num
"""
df_summary = con.execute(summary_query).df()

plt.figure(figsize=(10, 5))
plt.bar(df_summary['page_num'], df_summary['line_count'], color='skyblue', edgecolor='black')
plt.title('Extracted Line Images per Page')
plt.xlabel('Page Number')
plt.ylabel('Line Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#------

import matplotlib.pyplot as plt
from PIL import Image
import re

# 1. Query for the filename of the last line on the first page
query_line = """
WITH parsed_files AS (
    SELECT 
        folder,
        filename,
        -- Matches the full number after 'p'
        TRY_CAST(regexp_extract(filename, 'p([0-9]+)', 1) AS INT) AS page_num,
        -- Matches the full number after 'line'
        TRY_CAST(regexp_extract(filename, 'line([0-9]+)', 1) AS INT) AS line_num
    FROM raw_files
    WHERE folder = 'lineimages'
)
SELECT folder, filename, page_num, line_num
FROM parsed_files
WHERE page_num = (SELECT MIN(page_num) FROM parsed_files WHERE page_num IS NOT NULL)
ORDER BY line_num DESC
LIMIT 1 OFFSET 5;
"""

result= con.execute(query_line).fetchone()

# plot the lineimage
if result:
    folder, raw_filename, page_num, line_num = result
    
    # Strip ANSI escape codes to restore the clean filename
    clean_filename = re.sub(r'\x1b\[[0-9;]*m', '', raw_filename)
    
    # Construct image path
    image_path = os.path.join(".", folder, clean_filename)

    # Display image
    if os.path.exists(image_path):
        img = Image.open(image_path)
        
        plt.figure(figsize=(12, 3))
        plt.imshow(img)
        plt.title(f"Page {page_num} — Line {line_num}: {clean_filename}", fontsize=12)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"File not found: {image_path}")
    
    
#########---- all the line from a page
# 1. Clean ANSI escape codes from filenames in DuckDB (if not done already)
con.execute("""
    UPDATE raw_files 
    SET filename = regexp_replace(filename, '\x1b\[[0-9;]*m', '', 'g');
""")

# 2. Query all lines for page 2 sorted by line_num ascending
query_page_2 = """
WITH parsed_files AS (
    SELECT 
        folder,
        filename,
        TRY_CAST(regexp_extract(filename, 'p([0-9]+)', 1) AS INT) AS page_num,
        TRY_CAST(regexp_extract(filename, 'line([0-9]+)', 1) AS INT) AS line_num
    FROM raw_files
    WHERE folder = 'lineimages'
)
SELECT folder, filename, page_num, line_num
FROM parsed_files
WHERE page_num = 2
ORDER BY line_num ASC;
"""

lines = con.execute(query_page_2).fetchall()

# 3. Loop through and plot each line
for folder, raw_filename, page_num, line_num in lines:
    # Ensure ANSI codes are removed
    clean_filename = re.sub(r'\x1b\[[0-9;]*m', '', raw_filename)
    image_path = os.path.join(".", folder, clean_filename)

    if os.path.exists(image_path):
        img = Image.open(image_path)
        
        plt.figure(figsize=(12, 2))
        plt.imshow(img)
        plt.title(f"Page {page_num} — Line {line_num} ({clean_filename})", fontsize=10)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Skipping missing file: {image_path}")
        
        
