import os
import random
import re
import sys
import numpy
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages

#done this!
def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    NumLinks = len(corpus[page])
    corpus_of_page = corpus[page]
    corpus_dict = {}
    #print(corpus_of_page)
    if corpus_of_page == set():
        for link in corpus:
            corpus_dict[link] = (1/float(len(corpus)))
        return corpus_dict
    #for damping_factor d
    for link in corpus_of_page:
        corpus_dict[link] = (damping_factor/float(NumLinks))# + 
    tot = 0
    #for damping_factor 1-d
    for link in corpus:
        try:
            corpus_dict[link] += ((1-damping_factor)/float(len(corpus)))
        except KeyError:
            corpus_dict[link] = 0
            corpus_dict[link] += ((1-damping_factor)/float(len(corpus)))
    for link in corpus_dict:
        tot += corpus_dict[link]
    #print("total = ", tot)
    return corpus_dict

#done this!
def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    all_pages = []
    for page in corpus:
        all_pages.append(page)
    page = all_pages[random.randrange(len(all_pages))]
    
    page_account = dict.fromkeys(corpus.keys(), 0)

    page_account[page] += 1
    proba_corpus = {}
    for j in corpus:
        proba_corpus[j] = transition_model(corpus, j, damping_factor)
    for i in range(n-1):
        keys = [*proba_corpus[page].keys()]
        values = [*proba_corpus[page].values()]
        page = numpy.random.choice(keys, p=values)
        page_account[page] += 1
    for k in corpus:
        page_account[k] /= n
    return page_account
def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pagerank_dict = dict.fromkeys(corpus.keys(), 1.0/N)
    change = 0
    count = 0
    while True:
        count += 1
        new_pagerank_dict = dict.fromkeys(corpus.keys(), 0)
        for i in corpus:
            num_links_i = len(corpus[i])

            if num_links_i != 0:
                for p in corpus[i]:
                    new_pagerank_dict[p] += (damping_factor*pagerank_dict[i]/num_links_i)

            else:
                for page in corpus:
                    new_pagerank_dict[page] += (damping_factor*pagerank_dict[i]/N)

            new_pagerank_dict[i] += (1-damping_factor)/N
        changed = False
        for key in corpus:
            if new_pagerank_dict[key] - pagerank_dict[key] > 0.001:
                changed = True
                break
        pagerank_dict = copy.deepcopy(new_pagerank_dict)
        if changed == False:
            break

    return pagerank_dict

if __name__ == "__main__":
    main()
