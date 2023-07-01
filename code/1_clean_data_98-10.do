log using "${log}\1_clean_data_98-10.smcl", replace
use "${rdata}\mu1991to2020.dta", clear
label values earn

* 1997、2007 年最低工資改變，使用 2000 - 2007 的數據來做 Train
drop if year < 1998 | year > 2010

* 更改性別編號女為0
gen gender = 0 if sex == 2
replace gender = 1 if sex == 1
* 更改婚姻狀況 未婚1：0、離婚、分居3：0、配偶死亡4:0、有配偶(含與人同居)2：1
gen marital = 0 if mar == 1 | mar == 3 | mar == 4
replace marital = 1 if mar == 2


* 是否住臺北：63
gen taipei = 0
replace taipei = 1 if county == 63

* 薪資/工時
replace earn = 0 if earn == 1
gen earn_hr = earn/(hour*4)
replace earn_hr = 0 if earn_hr == .
gen earn_ad_hr = earn_ad/(hour*4)
replace earn_ad_hr = 0 if earn_ad_hr == .

* 篩選變數
keep year taipei age gender marital edu eduyr kid hour earn earn_ad earn_hr earn_ad_hr


save "${wdata}\mu_98-10", replace
log close