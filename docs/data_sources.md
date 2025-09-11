# Data Sources

## News Data

#### <u>News API Aggregators</u>
- thenewsapi.com
   - 100 daily request
   - 3 articles / news request
- alphavantage.co
   - 25 daily requests
- GNews
   - 100 daily request
   - 1 request per second
   - Up to 10 articles per request
   - 12 hr delay
   - 30 days historical data
- newsapiorg
   - 100 daily requests
   - 24 hr delay
   - 1 month historical data

#### <u>Direct Sources</u>
- The New York Times
   - Offers a limited free API for non-commercial use.
- The Guardian
   - Free "developer key" for non-commercial projects with usage limits. 
- yfinance web scraping
- Associated Press Webscraping
- https://www.sec.gov/search-filings
- https://seekingalpha.com/earnings/earnings-call-transcript
- **Reddit/Twitter** (if we have time)
   - snscrape (a Twitter scraping library)
   - PRAW (access to Reddit API)

## Stock Data

- WRDS Database (CRSP)
   - Daily Data
   - Relevant corporate actions to adjust close prices

- Investment Universe
   - 50 - 100 stocks in S&P500 with highest liquidity? or some other metric

## Sentiment Model
- FinBERT (https://huggingface.co/ProsusAI/finbert)
   - FinancialBERT (https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis)
      - Fine-tuned on Financial PhraseBank dataset
- FinABSA (https://huggingface.co/amphora/FinABSA)

