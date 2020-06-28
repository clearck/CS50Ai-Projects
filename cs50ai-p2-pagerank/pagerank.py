import os
import random
import re
import sys

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


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    # Get the entry for the current page:
    curr_page_links = corpus[page]
    n_links = len(curr_page_links)

    p = {}

    # If the current site has no outgoing links, we chose randomly amongst all pages with
    # equal probability
    if n_links == 0:
        for page in corpus.keys():
            p_even = 1 / len(corpus)
            p.update({page: p_even})
    else:
        # Probability to reach another site by randomly choosing it.
        p_random_page = (1 - damping_factor) / len(corpus)

        # Probability ro reach another site by clicking on one of the links on the current site.
        p_random_link = damping_factor / n_links

        for site in corpus:
            if site in curr_page_links:
                p.update({site: p_random_link + p_random_page})
            else:
                p.update({site: p_random_page})
    return p


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Start with a randomly selected page
    starting_page = random.choice(list(corpus.keys()))
    t_model = transition_model(corpus, starting_page, damping_factor)

    random_pages = []
    x = 0
    while x < n:
        # Select a random page out of the previous transition model
        weights = list(t_model.values())
        random_page = random.choices(list(t_model.keys()), weights=weights)[0]
        random_pages.append(random_page)

        # Generate a new transition model for a new sample
        t_model = transition_model(corpus, random_page, damping_factor)
        x += 1

    # Evaluate samples
    ranking = {}
    for site in list(corpus.keys()):
        ranking.update({site: random_pages.count(site) / n})
    return ranking


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    n = len(corpus)
    page_ranks = {}

    # Initialize starting probabilities
    for page in corpus.keys():
        page_ranks.update({page: 1 / n})

    previous_page_rank = 1
    current_page_rank = 0
    while abs(previous_page_rank - current_page_rank) > 0.001:
        for p in page_ranks.keys():
            # Sum over the page ranks from each possible page i that links to page p
            sigma = sum(page_ranks[i] / len(corpus[i]) for i in page_ranks.keys() if p in corpus[i])

            previous_page_rank = page_ranks[p]

            # Update current page rank with provided formula
            current_page_rank = (1 - damping_factor) / n + damping_factor * sigma
            page_ranks[p] = current_page_rank

    return page_ranks


if __name__ == "__main__":
    main()
