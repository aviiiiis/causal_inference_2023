log using "${log}\3_gen_prediction_data.smcl", replace

use "${rdata}\mu1991to2020.dta", clear
label values earn

* 使用 2000-2010 的數據來預測
drop if year < 2000 | year > 2010

* no observation
drop kid611
drop kid1214

* 去掉最高薪
drop if earn == 999999

* 剃除 >65 and <15
drop if age > 65 | age < 15

* 刪除工時大於周時數168
drop if hour > 168

* 刪除工時為0但月薪為正的人 以及 月薪為正但工時為0的人
drop if hour == 0 & earn > 0
drop if earn == 0 & hour > 0

* 更改性別編號女為0、男性為1
gen gender = 0 if sex == 2
replace gender = 1 if sex == 1

* 教育變數
tabulate edu, generate(edu)

* 家庭角色
tabulate rel, generate(rel)

* 婚姻變數
tabulate mar, generate(mar)

* 工作變數
tabulate work, generate(work)

* 縣市變數
tabulate county, generate(county)

* 科系變數
tab major, generate(major)

* 更改婚姻狀況 未婚1；離婚、分居3；配偶死亡4；有配偶(含與人同居)2
gen marital = 0 if mar == 1 | mar == 3 | mar == 4
replace marital = 1 if mar == 2

* 是否為省轄市、直轄市變數 基隆：17、臺北：63、臺中：66、臺南：67、高雄：64、新竹：18、嘉義：20
gen taipei = 0
replace taipei = 1 if county == 63

* 薪資/工時
replace earn = 0 if earn == 1
gen earn_hr = earn/(hour*4)
replace earn_hr = 0 if earn_hr == .
gen earn_ad_hr = earn_ad/(hour*4)
replace earn_ad_hr = 0 if earn_ad_hr == .


gen treat = 0
replace treat = 1 if earn_hr >= 95*1.25 & earn_hr < 95*1.75
replace treat = 2 if earn_hr >= 95*1.75
replace treat = . if earn_hr == 0


* 篩選變數
keep earn earn_ad year work* rel* mar* county* major* age gender edu* eduyr taipei marital kid* earn earn_ad hour earn_hr earn_ad_hr treat

label drop _all
save "${wdata}\mu_PD", replace

log close