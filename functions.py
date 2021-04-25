def scrape_front_page(url):
    import requests
    from bs4 import BeautifulSoup

    # Build empty dictionary to store results
    frontPageUrl_list = []

    # Get soup
    try:
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')

        # for all headings in soup store the heading as dict key and url as dict value,
        # count links found and print and headings not containing links
        headings = soup.find_all('h3')
        count = 0
        for i, heading in enumerate(headings):
            try:
                frontPageUrl_list.append(soup.find_all('h3')[i].find('a').get('href'))
                count += 1

            except:
                pass

    except:
        pass

    return frontPageUrl_list


def named_entities(sentence):
    from stanfordcorenlp import StanfordCoreNLP

    stopWords = []

    parser = StanfordCoreNLP('http://localhost', port=9000)

    namedEntities = parser.ner(sentence)
    toCheck = []
    hasNe = False
    matchingCorpus = []

    for i in range(len(namedEntities)):

        #check if entry is a named entity
        if namedEntities[i][1] == 'ORGANIZATION':
            matchingCorpus.append(namedEntities[i][0])
            if namedEntities[i][0] not in stopWords:
                hasNe = True
                toCheck.append(namedEntities[i][0])


    return hasNe, toCheck, matchingCorpus


def check_if_asx(toCheck, asxCompanies):
    from fuzzywuzzy import fuzz

    hasAsx = False
    asxTag = []
    companyName = []
    matchedEntity = []

    for i in range(len(asxCompanies['Company name'])):
        for entity in toCheck:

            # Use FuzzyWuzzy to check if tokens that appear in both named Entity and ASX Company name are very similar
            if fuzz.token_set_ratio(asxCompanies['Company name'][i], entity) > 95:
                if asxCompanies['ASX code'][i] not in asxTag:
                    asxTag.append(asxCompanies['ASX code'][i])
                    companyName = asxCompanies['Company name'][i]
                    matchedEntity = entity
                    hasAsx = True

                    # check if more than one company identfied and if so assign no tag as sentiment will not be clearly assigned
    if len(asxTag) != 1:
        hasAsx = False
        asxTag = []
        companyName = []
        matchedEntity = []


    return hasAsx, asxTag, companyName, matchedEntity


def get_sentiment(text):
    import flair

    try:
        sentimentModel = flair.models.TextClassifier.load('en-sentiment')

        sentence = flair.data.Sentence(text)
        sentimentModel.predict(sentence)
        if sentence.labels[0].value == "POSITIVE":
            sentiment = float(sentence.labels[0].score)

        else:
            sentiment = -1 * float(sentence.labels[0].score)

    except:
        sentiment = "NaN"

    return sentiment
