from bs4 import BeautifulSoup
import requests
from time import sleep
from random import randint
import pandas as pd
import os
import io


encoding = 'ISO-8859-1'


def soup_scrape(url):
    base_dir = "page_cache/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    url_hash = url.replace("/", "").replace(":", "").replace("?", "").replace(".", "")
    try:
        with open(base_dir+url_hash, "r") as file_read:
            return BeautifulSoup(file_read.read(), "html.parser")
    except Exception:
        print(url)
        sleep(randint(2, 7))
        html_data = requests.get(url,
                                 headers={"User-Agent":
                                              "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, "
                                              "like Gecko) Chrome/58.0.3029.110 Safari/537.36"}).text
        soup_data = BeautifulSoup(html_data, "html.parser")
        with open(base_dir+url_hash, "w") as file_write:
            file_write.write(html_data)
        return soup_data
    except RuntimeError:
        print(RuntimeError)


def update_results(years, weeks,
                   previous_dfs_results,
                   prev_results):
    if 'DK' in previous_dfs_results:
        base_url = "http://rotoguru1.com/cgi-bin/fyday.pl?game=dk&scsv=1&week=WEEK&year=YEAR"
    else:
        base_url = "http://rotoguru1.com/cgi-bin/fyday.pl?game=fd&scsv=1&week=WEEK&year=YEAR"

    if prev_results is False:
        prev_dfs_results_df = pd.DataFrame()
    else:
        prev_dfs_results_df = pd.read_csv(previous_dfs_results)

    all_games = pd.DataFrame()
    for yr in years:
        for wk in weeks:
            soups = soup_scrape(base_url.replace("WEEK", wk).replace("YEAR", yr))
            all_games = pd.concat([all_games, pd.read_csv(io.StringIO(soups.find("pre").text), sep=";")])

    prev_dfs_results_df = pd.concat((prev_dfs_results_df, all_games)).drop_duplicates().drop_duplicates()
    prev_dfs_results_df.to_csv(previous_dfs_results, index=False)

    return prev_dfs_results_df


if __name__ == '__main__':

    import pandas as pd

    #  ['12'] #list(map(str,range(1,4))) # 1,18)
    WEEKS = list(map(str, range(1, 19)))
    #  ['2017]
    YEARS = list(map(str, range(2011, 2018)))

    previous_DFS_DK_Results = 'DFS_NFL_Salaries/DFS_DK_Historical_Salary.csv'
    previous_DFS_FD_Results = 'DFS_NFL_Salaries/DFS_FD_Historical_Salary.csv'
    DFS_historical_Results = update_results(YEARS, WEEKS,
                                            previous_DFS_FD_Results,
                                            prev_results=False)
