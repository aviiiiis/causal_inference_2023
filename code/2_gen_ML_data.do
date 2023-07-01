log using "${log}\2_gen_ML_data.smcl", replace
use "${rdata}\mu1991to2020.dta", clear

* 1997、2007 年最低工資改變，使用 1998 - 2007 的數據來做 Train
drop if year < 1998 | year > 2007

* no observation
drop kid611
drop kid1214

* 去掉沒有工資或虧損的人
drop if earn  == 0 | earn  == 1
* 去掉最高薪
drop if earn == 999999
* 刪除工時為0但月薪為正的人 以及 月薪為正但工時為0的人
drop if hour == 0 & earn > 0
drop if earn == 0 & hour > 0

* 剃除 >65 and <15
drop if age > 65 | age < 15

* 受私人、政府僱用
keep if stat1 == 3 | stat1 == 4

* 更改性別編號女為0、男性為1
gen gender = 0 if sex == 2
replace gender = 1 if sex == 1


* 刪除工時大於周時數168
drop if hour > 168

* 教育變數
tabulate edu, generate(edu)

* 婚姻變數
tabulate mar, generate(mar)

* 工作變數
tabulate work, generate(work)

* 縣市變數
tabulate county, generate(county)

* 科系變數
tab major, generate(major)

* 家庭地位
tabulate rel, generate(rel)

* 更改婚姻狀況 未婚1；離婚、分居3；配偶死亡4；有配偶(含與人同居)2
gen marital = 0 if mar == 1 | mar == 3 | mar == 4
replace marital = 1 if mar == 2

* 是否住臺北：63
gen taipei = 0
replace taipei = 1 if county == 63

* 時薪 = 工資/工時
gen earn_hr = earn/(hour*4)
gen earn_ad_hr = earn_ad/(hour*4)
/*
* 2007年修改最低工資為 95 ，視低於125%的人為實驗組
gen treat_sec = 0
replace treat_sec = 1 if earn_hr >= 95*1.25 & earn_hr < 95*1.75
replace treat_sec = 2 if earn_hr >= 95*1.75

gen treat_fir = 0
replace treat_fir = 1 if earn_hr >= 95*1.75
*/

* 2007年修改最低工資為 17280 ，視低於125%的人為實驗組
gen treat = 0
replace treat = 1 if earn >= 17280*1.2 & earn < 17280*2
replace treat = 2 if earn >= 17280*2



* 篩選變數
keep year age gender edu* eduyr taipei marital kid* work* rel* mar* county* major* earn earn_ad earn_hr earn_ad_hr hour treat*

label drop _all
save "${wdata}\mu_ML", replace

log close