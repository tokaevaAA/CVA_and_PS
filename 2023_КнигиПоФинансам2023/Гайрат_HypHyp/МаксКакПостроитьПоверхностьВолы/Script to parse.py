# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import ps, ps_utils

import datetime as dt
from datetime import timedelta
from typing import Union, Optional, List
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as plty

import schedule
import time
import datetime as dt

import win32com.client as win32com
# from win32com.client import constants
#import win32clipboard
import pandas as pd
from datetime import timedelta
import shutil

#import xlwt

ps_utils.switch_to_dev()


ps_utils.switch_to_dev()
algo_hyphyp = '3fcf99221a2d86e1d669b07ec3859160f4e5d36d'


algo_id_Quotes = 'c39e37ce2bf31f932ee99bb8d37647358a24fc86'
AlgoIDConfig = 'bf6928ffa900025b25fd0eb77c4347e961547739'
BidAskRules = '0cfc1126b8f9d78e72991aac33cc3449237db3f3'
# b625d85149c5e9d49560bb8ff9dd66be8f3a084f

#price_dt = dt.date(2022, 11, 30)
price_dt = dt.datetime.today().date()

if price_dt < dt.datetime.today().date():
    used_mds = ps_utils.get_mds(price_dt)
else:
    used_mds = ps_utils.get_mds()

SDS = ps_utils.env.get_sds()


Holidays = pd.DataFrame(SDS.HolidayCalendars.MWB.ExceptionalHolidays)
Holidays[0] = Holidays[0].dt.strftime('%Y-%m-%d')
Holidays = list(Holidays[0])

FX_FWDs_Curve_Dict = {}
FX_FWDs_Curve_Dict['CNHRUB'] = ['CNH_FX', 'RUB_SOFR']
FX_FWDs_Curve_Dict['USDRUB'] = ['USD_XCCY', 'RUB_RUONIA_OIS']


ExcelApp = win32com.Dispatch("Excel.Application")
ExcelApp.Visible = True
# ExcelApp.CutCopyMode = False

# Open the desired workbook

# workbook = ExcelApp.Workbooks.Open(r"\\Bordo101.sigma.sbrf.ru\box2\GM_Trading\Rates\Structured\MD Parser/HW MD Parser v2.xlsx")

schedule.clear()




def FX_Forward_Curve(FX_Pair, Hist_period = 'Now' , Tenors = ['1M', '3M', '6M', '9M','1Y', '2Y', '3Y', '4Y', '5Y']):
    Projection_YC_Curr1 = FX_FWDs_Curve_Dict[FX_Pair][0]
    Projection_YC_Curr2 = FX_FWDs_Curve_Dict[FX_Pair][1]
    Pair = FX_Pair
    MarketDataFeed = ps_utils.env.get_root().MarketDataFeed
    
    if Hist_period == 0:
        df2 = pd.DataFrame(np.nan, index=[dt.datetime.strftime(dt.datetime.today().date(), '%Y-%m-%d')], columns=(['Spot'] + Tenors))
        used_mds = MarketDataFeed.LatestMarketDataSet
        Spot = used_mds.Spots[Pair].Value
        Pillar = ps_utils.analytics.get_dates_from_shifters(dt.datetime.today().date(), Tenors ,'MWB','ModifiedFollowing') 
        FWD = Spot * np.array(ps_utils.get_analytic_discounts(Projection_YC_Curr1, Pillar, used_mds)) / np.array(ps_utils.get_analytic_discounts(Projection_YC_Curr2, Pillar, used_mds))
        df2.loc[dt.datetime.strftime(dt.datetime.today().date(), '%Y-%m-%d')] = np.append( Spot, FWD)
        df2 = df2.dropna()
        df2 = df2.sort_index(ascending = False)
    else:
        Dates = list(MarketDataFeed.MarketData.keys())[-Hist_period:]

        df2 = pd.DataFrame(np.nan, index=Dates, columns=(['Spot'] + Tenors))

        for Date in Dates:
            if Date not in Holidays:
                DTDate = dt.datetime.strptime(Date, '%Y-%m-%d')
                used_mds = MarketDataFeed.MarketData[Date]
                Spot = used_mds.Spots[Pair].Value
                Pillar = ps_utils.analytics.get_dates_from_shifters(DTDate, Tenors ,'MWB','ModifiedFollowing') 
                #tenor10Y = dt.datetime(2032, 11, 23)
                FWD = Spot * np.array(ps_utils.get_analytic_discounts(Projection_YC_Curr1, Pillar, used_mds)) / np.array(ps_utils.get_analytic_discounts(Projection_YC_Curr2, Pillar, used_mds))
                df2.loc[Date] = np.append( Spot, FWD)
                df2 = df2.dropna()
                df2 = df2.sort_index(ascending = False)
        df2 = df2.sort_index(ascending = False)
    return df2

def IR_Forward_Curve(Curve, Hist_period, Change_sds = False, Tenors = ['1M', '2M', '3M', '6M', '9M','1Y', '2Y', '3Y', '4Y', '5Y', '6Y', '7Y', '8Y', '9Y', '10Y']):

    MarketDataFeed = ps_utils.env.get_root().MarketDataFeed
    
    SDS = ps_utils.env.get_sds()

    # MarketDataFeed.StaticDataSetID = ps.put(MarketDataFeed.StaticDataSetID)

    RateToGet = Curve
    RatesDict = {}
    
    Test = pd.DataFrame( columns = ['AsOf', 'Tenor', 'FWD'] )
    
    if Change_sds == True:
        SDS.Indices.RUB_RUONIA_OIS.Period = '1D'
        SDS.Indices.RUB_RUONIA_OIS.Type = 'IBOR'  
        
    if Hist_period >0:
    
        Dates = list(MarketDataFeed.MarketData.keys())[-Hist_period:]



        for Date in Dates:
            DTDate = dt.datetime.strptime(Date, '%Y-%m-%d')
            Pillars = ps_utils.analytics.get_dates_from_shifters(DTDate, Tenors ,'MWB','ModifiedFollowing') 

            RS = MarketDataFeed.MarketData[Date].RatesCurvesBundles[RateToGet].RatesSchedule

            #MarketDataFeed.MarketData[Date].RatesCurvesBundles[RateToGet].InterpolationMethod = 'Linear'
            #MarketDataFeed.MarketData[Date].RatesCurvesBundles[RateToGet].ExtrapolationMethod = 'Linear'
            #MarketDataFeed.MarketData[Date].RatesCurvesBundles[RateToGet].InterpolationSpaceCoordinates = 'LogarithmOfDiscountFactors'




            FWDRates = ps_utils.get_analytic_forwards(RateToGet, Pillars, MarketDataFeed.MarketData[Date], sds = SDS)
            RatesDict[Date] = np.array( [Tenors, FWDRates])

        
    else:
        
        Dates = [dt.datetime.strftime(dt.datetime.today().date(), '%Y-%m-%d')]
        
        Pillars = ps_utils.analytics.get_dates_from_shifters(dt.datetime.today().date(), Tenors ,'MWB','ModifiedFollowing')
        
        FWDRates = ps_utils.get_analytic_forwards(RateToGet, Pillars, MarketDataFeed.LatestMarketDataSet, sds = SDS)
        RatesDict[Dates[0]] = np.array( [Tenors, FWDRates])
        
        
    for Date in Dates:
        RatesDict[Date]
        A = pd.DataFrame( np.vstack( [[Date]* len(RatesDict[Date][0]), RatesDict[Date]]).transpose(), columns = ['AsOf', 'Tenor', 'FWD'] )
        Test = pd.concat([Test, A], ignore_index = True)


    Test['FWD'] = Test['FWD'].apply( float) * 100    

    Test['Tenor'] = Test['Tenor'].apply( str)

    order = Tenors
    CurveHist = Test[['AsOf', 'Tenor', 'FWD']].pivot( index = 'AsOf', columns = 'Tenor', values = 'FWD')[order].sort_index(ascending = False)
    return CurveHist


from ctypes import windll


#win32clipboard.EmptyClipboard() 
#win32clipboard.EmptyClipboard() 
def FX_for_schedule():
    try:
        if dt.datetime.strftime(dt.datetime.today().date(), '%Y-%m-%d') not in Holidays:
            if windll.user32.OpenClipboard(None):
                windll.user32.EmptyClipboard()
                windll.user32.CloseClipboard()
                
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("FX Spots").Cells(3,1).EntireRow.Insert()
        
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("MD").Range("J2:J4").Calculate()
            b = ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("MD").Range("J2:J4").Value
            #ExcelApp.Sheets("IR").Range("A2").PasteSpecial()
            FX = [i[0] for i in b]
        
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("FX Spots").Range("B3:D3").Value = FX
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("FX Spots").Range("A3").Value = dt.datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=3)
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Save()
    except:
        print( "Chto-to poshlo ne tak v "+ str(dt.datetime.now()))
    
def IR_for_schedule():
    try:
        if dt.datetime.strftime(dt.datetime.today().date(), '%Y-%m-%d') not in Holidays:
            if windll.user32.OpenClipboard(None):
                windll.user32.EmptyClipboard()
                windll.user32.CloseClipboard()
                
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("IR Curves").Cells(3,1).EntireRow.Insert()
        
        
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("MD").Range("D2:D50").Calculate()
            a = ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("MD").Range("D2:D50").Value
        
            #ExcelApp.Sheets("IR").Range("A2").PasteSpecial()
            IR = [i[0] for i in a]
        
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("IR Curves").Range("B3:AX3").Value = IR
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("IR Curves").Range("A3").Value = dt.datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=3)
        
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Save()
    except:
        print( "Chto-to poshlo ne tak v "+ str(dt.datetime.now()))
    
def FXIRFWDs_for_schedule_PS():
    
    if dt.datetime.strftime(dt.datetime.today().date(), '%Y-%m-%d') not in Holidays:
    
        if windll.user32.OpenClipboard(None):
            windll.user32.EmptyClipboard()
            windll.user32.CloseClipboard()
            
        ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("IR Fwds").Cells(3,1).EntireRow.Insert()
        ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("FX Fwds").Cells(3,1).EntireRow.Insert()
        
        FWDs_to_save = FX_Forward_Curve('CNHRUB', 1)
        IR_FWDS = IR_Forward_Curve('RUB_KEY_RATE', 1, True)
    
        rec_array = FWDs_to_save.to_records()
        rec_array_IR = IR_FWDS.to_records()
        rec_array = rec_array.tolist()
        rec_array_IR = rec_array_IR.tolist()
        ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("FX Fwds").Range("A3:K3").Value = rec_array
        ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("IR Fwds").Range("A3:P3").Value = rec_array_IR
        ExcelApp.Workbooks("HW MD Parser v2.xlsx").Save()


def Generate_Req(Product):
    
    
    CF_Req = ps.new_cm()
    CF_Req.RequestName = 'PricingUI'
    CF_Req.MarketDataSet = ps_utils.get_mds()
    
    #CF_Req.MarketDataSet = '5d3290592800ae8ca1bc6193fed3c26ae5894aeb'
    CF_Req.StaticDataSet = ps_utils.env.get_sds()
    CF_Req.AlgoIDConfig = AlgoIDConfig
    CF_Req.Product = Product
    CF_Req.BidAskRules = BidAskRules
    CF_Req.ReportingCurrency = 'RUB'

    Res = ps_utils.compute(ps.put(CF_Req), algo_id_Quotes)
    return Res

def Generate_Product(Option): 
    CF_Product = ps.new_cm()
    CF_Product.TypeName = 'Portfolio'
    CF_Product.CounterpartyCreditRating = 15
    CF_Product.CounterpartyLossGivenDefault = 0.75
    CF_Product.OwnCreditCurveIdentifier = 'RUB_SBER_OFZ_SNR_CR'
    CF_Product.Products = [Option]
    return CF_Product
def Generate_CF(Strike, Notional, Tenor,  Position = 'ClientBuys', PayoffType = 'Cap'):
    CF = ps.new_cm()
    CF.PayoffType = PayoffType
    CF.Position = Position
    CF.TypeName = 'InterestRatesCapFloorProduct'
    CF.PremiumIsRunning = 'false'
    CF.FloatingRateAveragingPeriod = '1D'
    Maturity = ps_utils.analytics.get_dates_from_shifters(price_dt, Tenor ,'MWB','ModifiedFollowing')
    start_dt = ps_utils.get_dates_from_shifters(price_dt, '1BD','MWB','Following')
    Schedule = ps_utils.get_schedule(start_dt,Maturity, '1M','MWB','ModifiedFollowing')
    CF.PeriodStartDates = Schedule[:-1]
    CF.PeriodEndDates = Schedule[1:]
    CF.PaymentDates = Schedule[1:]
    CF.Strike = Strike
    CF.Notionals = [Notional] * (len(Schedule)-1)
    CF.IndexIdentifier = 'RUB_KEY_RATE'
    CF.DayCountConvention = 'ActAct'
    CF.PaymentCurrency = 'RUB'
    CF.PaymentCalendar = 'MWB'
    CF.StartDate = Schedule[0]
    CF.EndDate = Schedule[-1]
    return CF

def QuotesForTerminal():
    
    
    Tenors = ['1Y', '3Y', '5Y']
    Strikes_C = [0.08, 0.1, 0.12]
    Strikes_F = [0.05, 0.06, 0.07]
    
    
    a = pd.DataFrame(columns = ['RATE NAME', 'TENOR', 'SIZE, BN RUB', 'CAP/FLOOR', 'STRIKE', 'PREMIUM BID, %', 'PREMIUM OFFER, %'])
    Dict_a = {}
    
    c = 0

    for i in Tenors:
        for j in Strikes_F:
            Dict_a = {'RATE NAME': 'key_rate_daily_average_monthly',
                      'TENOR': i,
                      'SIZE, BN RUB': 3,
                      'STRIKE': j,
                     'CAP/FLOOR': 'Floor'}
            a = pd.concat( [a,  pd.DataFrame(Dict_a, index = [c])], axis = 0)
            c = c + 1
            
    for i in Tenors:
        for j in Strikes_C:
            Dict_a = {'RATE NAME': 'key_rate_daily_average_monthly',
                      'TENOR': i,
                      'SIZE, BN RUB': 3,
                      'STRIKE': j,
                     'CAP/FLOOR': 'Cap'}
            a = pd.concat( [a,  pd.DataFrame(Dict_a, index = [c])], axis = 0)
            c = c + 1


    for i in range(len(a)):
        Res = Generate_Req(Generate_Product(Generate_CF(a.iloc[i]['STRIKE'], a.iloc[i]['SIZE, BN RUB'] * 10**9, a.iloc[i]['TENOR'], Position = 'ClientBuys', PayoffType = a.iloc[i]['CAP/FLOOR'])))
        a['PREMIUM OFFER, %'][i] =(Res.Price.Mid + Res.Price.TotalCharge )/(a.iloc[i]['SIZE, BN RUB'] * 10**9 )

        Res = Generate_Req(Generate_Product(Generate_CF(a.iloc[i]['STRIKE'], a.iloc[i]['SIZE, BN RUB'] * 10**9, a.iloc[i]['TENOR'], Position = 'ClientSells', PayoffType = a.iloc[i]['CAP/FLOOR'])))
        a['PREMIUM BID, %'][i] =  max((Res.Price.Mid - Res.Price.TotalCharge )/(a.iloc[i]['SIZE, BN RUB'] * 10**9), 0)
    
    
    Workbook = ExcelApp.Workbooks.Open(r'\\Bordo101.sigma.sbrf.ru\box2\GM_Trading\Rates\Structured\QuotesForTerminal/Interest rate options (IRO).xlsx')
    Workbook.Sheets('IRO').Range('A2:E19').Value = a[['RATE NAME', 'TENOR', 'SIZE, BN RUB', 'CAP/FLOOR', 'STRIKE']].to_records(index = False).tolist()
    Workbook.Sheets('IRO').Range('H2:I19').Value = a[['PREMIUM BID, %', 'PREMIUM OFFER, %']].to_records(index = False).tolist()
    Workbook.Save()
    Workbook.Close()
    if load_to_parser == True:
        try:
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("PricesForTerminal").Range('A2:E19').Value = a[['RATE NAME', 'TENOR', 'SIZE, BN RUB', 'CAP/FLOOR', 'STRIKE']].to_records(index = False).tolist()
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Sheets("PricesForTerminal").Range('H2:I19').Value = a[['PREMIUM BID, %', 'PREMIUM OFFER, %']].to_records(index = False).tolist()
            ExcelApp.Workbooks("HW MD Parser v2.xlsx").Save()
        except:
            pass
    print( "Quotes for Terminal are updated: " + str(dt.datetime.now()))


def CopyParser():
    try:
        src = r'\\Bordo101.sigma.sbrf.ru\box2\GM_Trading\Rates\Structured\MD Parser/HW MD Parser v2.xlsx'
        dst = r'\\Bordo101.sigma.sbrf.ru\box2\GM_Trading\Rates\Structured\MD Parser/CopyOfParser/ParserCopy.xlsx'
        shutil.copyfile(src, dst)
    except:
        try:
            src = r'\\Bordo101.sigma.sbrf.ru\box2\GM_Trading\Rates\Structured\MD Parser/HW MD Parser v2.xlsx'
            dst = r'\\Bordo101.sigma.sbrf.ru\box2\GM_Trading\Rates\Structured\MD Parser/CopyOfParser/ParserCopyBackup.xlsx'
            shutil.copyfile(src, dst)
        except:
            print("Someone opened the book")


schedule.clear()

schedule.every().day.at("09:00").do(FX_for_schedule)
schedule.every().day.at("10:00").do(FX_for_schedule)
schedule.every().day.at("11:00").do(FX_for_schedule)
schedule.every().day.at("12:00").do(FX_for_schedule)
schedule.every().day.at("13:00").do(FX_for_schedule)
schedule.every().day.at("14:00").do(FX_for_schedule)
schedule.every().day.at("15:00").do(FX_for_schedule)
schedule.every().day.at("16:00").do(FX_for_schedule)
schedule.every().day.at("17:00").do(FX_for_schedule)
schedule.every().day.at("18:00").do(FX_for_schedule)
schedule.every().day.at("19:00").do(FX_for_schedule)

schedule.every().day.at("09:00").do(IR_for_schedule)
schedule.every().day.at("10:00").do(IR_for_schedule)
schedule.every().day.at("11:00").do(IR_for_schedule)
schedule.every().day.at("12:00").do(IR_for_schedule)
schedule.every().day.at("13:00").do(IR_for_schedule)
schedule.every().day.at("14:00").do(IR_for_schedule)
schedule.every().day.at("15:00").do(IR_for_schedule)
schedule.every().day.at("16:00").do(IR_for_schedule)
schedule.every().day.at("17:00").do(IR_for_schedule)
schedule.every().day.at("18:00").do(IR_for_schedule)
schedule.every().day.at("19:00").do(IR_for_schedule)



#FX_for_schedule()
#IR_for_schedule()

schedule.every().day.at("11:00").do(FXIRFWDs_for_schedule_PS)

load_to_parser = True
schedule.every().day.at("09:40").do(QuotesForTerminal)
schedule.every().day.at("15:10").do(QuotesForTerminal)


schedule.every().day.at("09:00").do(CopyParser)
schedule.every().day.at("10:00").do(CopyParser)
schedule.every().day.at("11:00").do(CopyParser)
schedule.every().day.at("12:00").do(CopyParser)
schedule.every().day.at("13:00").do(CopyParser)
schedule.every().day.at("14:00").do(CopyParser)
schedule.every().day.at("15:00").do(CopyParser)
schedule.every().day.at("16:00").do(CopyParser)
schedule.every().day.at("17:00").do(CopyParser)
schedule.every().day.at("18:00").do(CopyParser)
schedule.every().day.at("19:00").do(CopyParser)



IR_for_schedule()
FX_for_schedule()
QuotesForTerminal()

while True:
 
    # Checks whether a scheduled task
    # is pending to run or not
    schedule.run_pending()
    # time.sleep(3550)
    time.sleep(1)

