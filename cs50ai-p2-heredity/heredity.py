import csv
import itertools
import sys
from functools import reduce

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """

    all_people = set(people.keys())
    people_info = map_arguments(all_people, one_gene, two_genes, have_trait)

    probabilities = {}

    for person in all_people:
        n_genes = people_info[person]['genes']
        father = people[person]['father']
        mother = people[person]['mother']

        # Check the probability, that a person has a specific amount of genes.
        # This assumes that a child always has 2 parents or none.
        if father == mother is None:
            p_genes = PROBS["gene"][n_genes]
        else:
            p_inherit_father = abs(0.5 * people_info[father]['genes'] - PROBS["mutation"])
            p_inherit_mother = abs(0.5 * people_info[mother]['genes'] - PROBS["mutation"])

            if n_genes == 0:
                # Person has no genes, meaning he/she doesn't get the from the father or the mother
                p_genes = (1 - p_inherit_father) * (1 - p_inherit_mother)
            elif n_genes == 1:
                # Person only has one gene, meaning he/she gets it either from his/her mother or father
                p_genes = p_inherit_father * (1 - p_inherit_mother) + (1 - p_inherit_father) * p_inherit_mother
            else:
                # Person has two genes, meaning he/she gets one from father and mother
                p_genes = p_inherit_father * p_inherit_mother

        # Check the probability, that a person has or doesn't have the trait
        p_trait = p_genes * PROBS["trait"][n_genes][people_info[person]['trait']]
        probabilities[person] = {'p_genes': p_genes, 'p_trait': p_trait}

    values = probabilities.values()
    p_traits = [entry['p_trait'] for entry in values]

    return reduce(lambda x, y: x * y, p_traits)


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """

    all_people = probabilities.keys()
    people_data = map_arguments(all_people, one_gene, two_genes, have_trait)

    for person in all_people:
        probabilities[person]['gene'][people_data[person]['genes']] += p
        probabilities[person]['trait'][people_data[person]['trait']] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """

    for person in probabilities.keys():
        for key in probabilities[person]:
            items = probabilities[person][key].items()
            values = [item[1] for item in items]
            s = sum(values)

            for k, v in items:
                probabilities[person][key][k] = v / s


def map_arguments(all_people, one_gene, two_genes, have_trait):
    """
    Takes all people and the group they are in and returns a mapping of
    name: #genes.
    """
    people_data = {}
    for person in all_people:
        has_trait = person in have_trait
        if person in one_gene:
            people_data[person] = {'genes': 1, 'trait': has_trait}
        elif person in two_genes:
            people_data[person] = {'genes': 2, 'trait': has_trait}
        else:
            people_data[person] = {'genes': 0, 'trait': has_trait}

    return people_data


if __name__ == "__main__":
    main()
