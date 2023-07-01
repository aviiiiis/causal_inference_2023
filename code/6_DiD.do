log using "${log}\6_DiD.smcl", replace
clear all
use "${wdata}\mu_MLPD.dta"



* Individual is predicated to be treated if the predicted probability is larger than 0.5.
replace treat = 2

replace treat = 1 if y_GBC == 0

replace treat = 0 if y_GBC == 1 

drop if treat == 2

* employment:1 Unemployment:0
gen employment = 0
replace employment = 1 if earn != 0

* The new minimum wage policy was implemented in July 2007, but MUS was done in May 2007.
gen post = 0
replace post = 1 if year >= 2008

* control variables
global xlist "age eduyr gender marital kid taipei"

drop if year < 2005
* interaction term in DiD
gen diff = post*treat

** Common Trend Test: mean comparison
* Test employment
egen mean_employment=mean(employment), by(year treat)
sum mean_employment if year == 2007 & treat == 0
gen base_mean_employment = r(mean)
sum mean_employment if year == 2007 & treat == 1
replace base_mean_employment = r(mean) if treat == 1
gen mean_employment_sd = mean_employment-base_mean_employment

graph twoway (connect mean_employment_sd year if treat==1,sort) (connect mean_employment_sd year if treat==0,sort lpattern(dash)), ///
xline(2007,lpattern(dash) lcolor(gray)) ///
ytitle("Employment") xtitle("Year") ///
ylabel(,labsize(*0.75)) xlabel(,labsize(*0.75)) ///
legend(label(1 "Treatment Group") label( 2 "Control Group")) ///
xlabel(2005 (1) 2010)  graphregion(color(white)) ///
name(mean_cp_employment, replace)
graph export "${pic}\mean_cp_employment_pred.png", replace

* Test working hour
egen mean_hour=mean(hour), by(year treat)
sum mean_hour if year == 2007 & treat == 0
gen base_mean_hour = r(mean)
sum mean_hour if year == 2007 & treat == 1
replace base_mean_hour = r(mean) if treat == 1
gen mean_hour_sd = mean_hour-base_mean_hour

graph twoway (connect mean_hour_sd year if treat==1,sort) (connect mean_hour_sd year if treat==0,sort lpattern(dash)), ///
xline(2007,lpattern(dash) lcolor(gray)) ///
ytitle("Working Hour") xtitle("Year") ///
ylabel(,labsize(*0.75)) xlabel(,labsize(*0.75)) ///
legend(label(1 "Treatment Group") label( 2 "Control Group")) ///
xlabel(2005 (1) 2010)  graphregion(color(white)) ///
name(mean_cp_hour, replace)
graph export "${pic}\mean_cp_hour_pred.png", replace

* Test adjusted earning
egen mean_earn_ad=mean(earn_ad), by(year treat)
sum mean_earn_ad if year == 2007 & treat == 0
gen base_mean_earn_ad = r(mean)
sum mean_earn_ad if year == 2007 & treat == 1
replace base_mean_earn_ad = r(mean) if treat == 1
gen mean_earn_ad_sd = mean_earn_ad-base_mean_earn_ad

graph twoway (connect mean_earn_ad_sd year if treat==1,sort) (connect mean_earn_ad_sd year if treat==0,sort lpattern(dash)), ///
xline(2007,lpattern(dash) lcolor(gray)) ///
ytitle("Adjusted Earning") xtitle("Year") ///
ylabel(,labsize(*0.75)) xlabel(,labsize(*0.75)) ///
legend(label(1 "Treatment Group") label( 2 "Control Group")) ///
xlabel(2005 (1) 2010)  graphregion(color(white)) ///
name(mean_cp_earning, replace)
graph export "${pic}\mean_cp_earn_pred.png", replace

** Common Trend Test: significance of interaction term between year and treat
gen policy = year - 2008

* Generate interaction term between year dummy and treat dummy
forvalues i = 3(-1)1{
  gen pre_`i' = (policy == -`i' & treat == 1) 
}

gen current = (policy == 0 & treat == 1)

forvalues j = 1(1)2{
  gen  post_`j' = (policy == `j' & treat == 1)
}

* Set pre_1(2007) as baseline year and regress
replace pre_1 = 0
drop if year > 2010
* Test employment
reg employment pre_* current post_* i.treat i.year $xlist 
* Visualization
coefplot, baselevels ///
keep(pre_* current post_*) ///
vertical ///
omitted ///
yline(0,lcolor(edkblue*0.8)) ///
xline(3, lwidth(vthin) lpattern(dash) lcolor(teal)) ///
ylabel(,labsize(*0.75)) xlabel(,labsize(*0.75)) ///
coeflabels(pre_3=2005 pre_2="2006" pre_1="2007" current= "2008" post_1="2009" post_2="2010") ///
ytitle("Policy Effect on Employment", size(small)) ///
xtitle("Year", size(small)) ///
addplot(line @b @at) /// add line between spots
ciopts(lpattern(dash) recast(rcap) msize(medium)) ///
graphregion(color(white)) ///
msymbol(circle_hollow)

graph rename pt_employment
graph export "${pic}\parallel_trend_test_employment_pred.png", replace

* Test working hour
reg hour pre_* current post_* i.treat i.year $xlist
* Visualization
coefplot, baselevels ///
keep(pre_* current post_*) ///
vertical ///
omitted ///
yline(0,lcolor(edkblue*0.8)) ///
xline(3, lwidth(vthin) lpattern(dash) lcolor(teal)) ///
ylabel(,labsize(*0.75)) xlabel(,labsize(*0.75)) ///
coeflabels(pre_3="2005" pre_2="2006" pre_1="2007" current= "2008" post_1="2009" post_2="2010") ///
ytitle("Policy Effect on Working Hour", size(small)) ///
xtitle("Year", size(small)) ///
addplot(line @b @at) /// add line between spots
ciopts(lpattern(dash) recast(rcap) msize(medium)) ///
graphregion(color(white)) ///
msymbol(circle_hollow)

graph rename pt_hour
graph export "${pic}\parallel_trend_test_hour_pred.png", replace

* Test adjusted earning
reg earn_ad pre_* current post_* i.treat i.year $xlist
* Visualization
coefplot, baselevels ///
keep(pre_* current post_*) ///
vertical ///
omitted ///
yline(0,lcolor(edkblue*0.8)) ///
xline(3, lwidth(vthin) lpattern(dash) lcolor(teal)) ///
ylabel(,labsize(*0.75)) xlabel(,labsize(*0.75)) ///
coeflabels(pre_3=2005 pre_2="2006" pre_1="2007" current= "2008" post_1="2009" post_2="2010") ///
ytitle("Policy Effect on Adjusted Earning", size(small)) ///
xtitle("Year", size(small)) ///
addplot(line @b @at) ///
ciopts(lpattern(dash) recast(rcap) msize(medium)) ///
graphregion(color(white)) ///
msymbol(circle_hollow)

graph rename pt_earn
graph export "${pic}\parallel_trend_test_earn_pred.png", replace

gen DiD = treat*post

rename hour Hour
rename employment Employment
rename earn_ad Earning
** DiD (repeated cross-sections)
reg Employment DiD i.treat i.year 
est sto Employment0
reg Hour DiD i.treat i.year if Employment == 1
est sto Hour0
reg Earning DiD i.treat i.year if Employment == 1
est sto Earning0
reg Employment DiD i.treat i.year $xlist 
est sto Employment1
reg Hour DiD i.treat i.year $xlist if Employment == 1
est sto Hour1
reg Earning DiD i.treat i.year $xlist if Employment == 1
est sto Earning1

esttab Employment0 Hour0 Earning0 Employment1 Hour1 Earning1 using "${table}\reg_pred.tex", keep(DiD $xlist) se tex replace
*esttab Employment1 Hour1 Earning1 using "${table}\reg.tex", keep(DiD $xlist) tex append
*esttab Employment0 Hour0 Earning0 Employment1 Hour1 Earning1, p

log close