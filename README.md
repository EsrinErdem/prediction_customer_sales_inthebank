# prediction_customer_sales_inthebank
Today, you started working in the Bank. Your manager take you to a meeting about a new commercial product of Bank. In this meeting, Mr. Yıldız, the Marketing Manager of the Commercial segment, stated that “Bank is one of the leading banks and we developed a new product for commercial clients. We have been working for months. As a result of the studies, we believe that clients with over 5 million TL annual net sales are the most suitable to offer this new product.”

He also added:  “As you know, banks do not have current financial statement of each company. Without financial statements, we cannot get proper net sales information of customers. We can already reach suitable clients for the new product for those with current financials available in our databases. However, we want to be sure that we reach all targeted customers. So, we have to find a way to decide which one of these clients without financials have actually over 5 million TL net sales.”

Then, the Commercial Loan Allocation Manager stated that total risk and total limit amounts in the banking sector can be reached for Bank clients and this information could be used for estimation.

Mr. Yıldız asked, “If the clients has more than 5 million TL total loan balance in banking sector, may  their net sales be higher than 5 million TL too?”

Commercial Loan Allocation Manager Mrs. İlhan replied “Even though there are exceptions, generally it is correct. In addition, companies with more experience and having institutional culture are generally in the leading position in the sector. Therefore, they should be evaluated."

Your manager took the floor and introduced you to other managers. “We can develop the most suitable model by looking at the data” he said, adding that you can develop a model that would solve this problem. In the end, by adding “with data we can create the most appropriate solution.”

After meeting, you take related information from databases. The following information could be reached through the queries in the databases. Information are shown as empty (NULL) if no data is available.

* YEAR	Date of data
* Customer_num	Customer identification number
* Establishment_Date	Company establishment date
* Number_of_Emp	Number of employees
* Profit	Annual profit
* Sector	Sector that company operates
* Region	Geographic region
* Total Risk 	Total loan balance amount in the banking sector
* Total Limit	Total limit in the banking sector
* Sales	0 if Sales =< 5 million TL
* 1 if Sales > 5 million TL
* 3 if Sales is not available.
