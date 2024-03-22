from markdown_crawler import md_crawl
url = 'https://scs.duytan.edu.vn/thong-bao-tuyen-sinh-chp/'
print(f'🕸️ Starting crawl of {url}')
md_crawl(
    url,
    max_depth = 3,
    num_threads=30,
    base_dir='programs',
    valid_paths=['//scs.duytan.edu.vn/'],
    is_domain_match=True,
    is_base_path_match=False,
    is_debug=True,
    is_links = True
)
