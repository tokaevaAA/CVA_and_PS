import ps_utils
from ps_utils.env import ps
import numpy as np
from ps_utils import env, market, analytics, pricing
import pandas as pd
from ps import message as msg
import sys



from typing import Union, Optional, Dict, List
PsObject = Union[ps.Message, ps.GlobalID]


def prepare_big_request_for_pnl(response, prev_response, algo_id: str = None, reporting_currency: str = None, queries: List[str] = None) -> msg.MessageList:

    if algo_id is None:
        algo_id = env.global_algo_id()

    big_request = ps.new_cm([])
    new_resp = ps.get(ps.put(response))
    new_prev_resp = ps.get(ps.put(prev_response))

    for r in new_resp:
        cpty = r.Request['_Counterparty']
        prev_r = next((x for x in new_prev_resp if x.Request['_Counterparty'] == cpty), None)
        req = ps.new_cm({
            'RequestName': 'Layer2',
            '_Counterparty': cpty,
            'Layer2Algo': algo_id,
            'ReportingCurrency': reporting_currency if reporting_currency else r.Request.ReportingCurrency,
            'Method': 'Revaluation',
            'Queries': queries or r.Request.Queries,
            'Request': r.Request,
            'Response': r,
        })
        if prev_r:
            new_prev_resp.remove(prev_r)
            req['PreviousRequest'] = prev_r.Request
            req['PreviousResponse'] = prev_r
        big_request.append(req)

    for r in new_prev_resp:
        cpty = r.Request['_Counterparty']
        big_request.append(ps.new_cm({
            'RequestName': 'Layer2',
            '_Counterparty': cpty,
            'Layer2Algo': algo_id,
            'ReportingCurrency': reporting_currency if reporting_currency else r.Request.ReportingCurrency,
            'Method': 'Revaluation',
            'Queries': queries or r.Request.Queries,
            'PreviousRequest': r.Request,
            'PreviousResponse': r
        }))

    return ps.new_cm(big_request)


def __calculate_pnl_category(request: msg.MessageList, prev_request: msg.MessageList, results: msg.MessageList, algo_id: str, category: str):
    # Calculating request
    calc_request = ps.compute_raw(algo_id, request).Result
    calc_request = [x for x in calc_request if x.get('StatusText') is not None and x.StatusText == 'done']
    calc_prev_request = ps.compute_raw(algo_id, prev_request).Result
    calc_prev_request = [x for x in calc_prev_request if x.get('StatusText') is not None and x.StatusText == 'done']

    # Matching results
    for res in calc_request:
        cpty = res.Request['_Counterparty']
        result = next((x for x in results if x.Request['_Counterparty'] == cpty))
        prev_res = next((x for x in calc_prev_request if x.Request['_Counterparty'] == cpty), None)
        for query in res.Request.Queries:
            result[query][category] = res.Result[query] - prev_res.Result[query] if prev_res else 0
        if prev_res is not None:
            calc_prev_request.remove(prev_res)

    for res in calc_prev_request:
        cpty = res.Request['_Counterparty']
        result = next((x for x in results if x.Request['_Counterparty'] == cpty))
        for query in res.Request.Queries:
            result[query][category] = 0

            
def __update_marketDependencies(res_market_dependencies: msg.MessageList, results: msg.MessageList, all_market_dependencies: dict, excluded_dependencies: dict):

    for res_dep in res_market_dependencies:
        cpty = res_dep.Request['_Counterparty']
        result = next(filter(lambda x: x.Request['_Counterparty'] == cpty, results))

        for Type in res_dep.Result.MarketDependencies:
            if Type == 'Fixings':
                continue

            for Identifier in res_dep.Result.MarketDependencies[Type]:
                if excluded_dependencies.get(Type) and Identifier in excluded_dependencies.get(Type):
                    continue

                md_to = result.Request.Request.Model.MarketDataSet[Type][Identifier]
                md_from = result.Request.PreviousRequest.Model.MarketDataSet[Type][Identifier]

                if isinstance(md_to, float):
                    if md_to != md_from:
                        result.MarketDependencies.setdefault(Type, [])
                        result.MarketDependencies[Type].append(Identifier)

                        all_market_dependencies.setdefault(Type, [])
                        if Identifier not in all_market_dependencies[Type]:
                            all_market_dependencies[Type].append(Identifier)
                elif md_to._id != md_from._id:
                    result.MarketDependencies.setdefault(Type, [])
                    result.MarketDependencies[Type].append(Identifier)

                    all_market_dependencies.setdefault(Type, [])
                    if Identifier not in all_market_dependencies[Type]:
                        all_market_dependencies[Type].append(Identifier)


def run_generic_revaluation_based_metric_change_explain_batch(big_request: msg.MessageList,
                                                              algo_id: str = None,
                                                              same_mds: bool = False,
                                                              threshold: float = -1,
                                                              optimize_credit_curves: bool = False) -> PsObject:
    """Generates revaluation based explanation report for the metrics requested.
    PS! PnL Explain is calculated from the same Base (compared to other methods)
    PPS! Make sure, input data is in correct format
    PPPS! One request per one cpty in big request!

    Args:
        big_request (MessageList): A list of by-counterparty collected requests on two compared dates
        algo_id (str): ID of the algo to use
        same_mds (bool): True if within date and within prev_date the same mds is assigned (improves performance)
        threshold: if Total change is less than threshold, then PnL Explain for given cpty is not calculated

    Returns:
        PsObject: revaluation based PnL explain summary
    """

    if algo_id is None:
        algo_id = env.global_algo_id()

    results = ps.new_cm([])

    env.logger.info('Calculating Total/Expired/New and removing failed cpties')

    request = ps.new_cm([])
    prev_request = ps.new_cm([])
    for req in big_request:
        if req.get('Request'):
            req.Request.Queries = req.Queries
            req.Request.ReportingCurrency = req.ReportingCurrency
            request.append(req.Request)
        if req.get('PreviousRequest'):
            req.PreviousRequest.Queries = req.Queries
            req.PreviousRequest.ReportingCurrency = req.ReportingCurrency
            prev_request.append(req.PreviousRequest)

    ps.put(request)
    ps.put(prev_request)

    env.logger.info(f'Requests collected. Request={request._id}. PrevRequest={prev_request._id}')
    response = ps.compute_raw(algo_id, request)
    prev_response = ps.compute_raw(algo_id, prev_request)

    for req in big_request:
        cpty = req['_Counterparty']
        result = ps.new_cm({
            '_Counterparty': cpty,
            'Request': req,
        })

        if not req.get('Request'):
            # Expired cpty
            prev_res = next(filter(lambda x: x.Request['_Counterparty'] == cpty, prev_response.Result), None)
            if prev_res.get('StatusText') and prev_res.get('StatusText') == 'done':
                result.update({
                    query: ps.new_cm({
                        'Total': -req.PreviousResponse.Result[query],
                        'ExpiredCounterparty': -req.PreviousResponse.Result[query]
                    }) for query in req.Queries
                })
            else:
                result.update({
                    query: ps.new_cm({'Total': 0}) for query in req.Queries
                })
            result['_NeedsToBeCalculated'] = False
        elif not req.get('PreviousRequest'):
            # New cpty
            res = next(filter(lambda x: x.Request['_Counterparty'] == cpty, response.Result), None)
            if res.get('StatusText') and res.get('StatusText') == 'done':
                result.update({
                    query: ps.new_cm({
                        'Total': req.Response.Result[query],
                        'NewCounterparty': req.Response.Result[query]
                    }) for query in req.Queries
                })
            else:
                result.update({
                    query: ps.new_cm({'Total': 0}) for query in req.Queries
                })
            result['_NeedsToBeCalculated'] = False
        else:
            res = next(filter(lambda x: x.Request['_Counterparty'] == cpty, response.Result), None)
            prev_res = next(filter(lambda x: x.Request['_Counterparty'] == cpty, prev_response.Result), None)

            if res.get('StatusText') and res.get('StatusText') == 'done' and prev_res.get('StatusText') and prev_res.get('StatusText') == 'done':
                result.update({
                    query: ps.new_cm({
                        'Total': req.Response.Result[query] - req.PreviousResponse.Result[query]
                    }) for query in req.Queries
                })
                result['_NeedsToBeCalculated'] = any([abs(result[query]['Total']) > threshold for query in req.Queries])
            elif res.get('StatusText') and res.get('StatusText') == 'done':
                result.update({
                    query: ps.new_cm({
                        'Total': req.Response.Result[query],
                        'ErrorPrevious': req.Response.Result[query]
                    }) for query in req.Queries
                })
                result['_NeedsToBeCalculated'] = False
            elif prev_res.get('StatusText') and prev_res.get('StatusText') == 'done':
                result.update({
                    query: ps.new_cm({
                        'Total': -req.PreviousResponse.Result[query],
                        'ErrorCurrent': -req.PreviousResponse.Result[query]
                    }) for query in req.Queries
                })
                result['_NeedsToBeCalculated'] = False
            else:
                result.update({
                    query: ps.new_cm({
                        'Total': 0,
                        'ErrorBoth': 0
                    }) for query in req.Queries
                })
                result['_NeedsToBeCalculated'] = False

        results.append(result)

    env.logger.info('Total calculated')

    env.logger.info('Doing some preparations')

    env.logger.info('Converting market data to canonical format')
    # Converting market data to canonical format
    if same_mds:
        mds_to = market.to_canonical_format(next(filter(lambda x: x['_NeedsToBeCalculated'], results), None).Request.Request.Model.MarketDataSet, algo_id)
        mds_from = market.to_canonical_format(next(filter(lambda x: x['_NeedsToBeCalculated'], results), None).Request.PreviousRequest.Model.MarketDataSet, algo_id)
        for request in big_request:
            if request.get('Request', None) is not None:
                request.Request.Model.MarketDataSet = mds_to
            if request.get('PreviousRequest', None) is not None:
                request.PreviousRequest.Model.MarketDataSet = mds_from
    else:
        for result in results:
            result.Request.Request.Model.MarketDataSet = market.to_canonical_format(result.Request.Request.Model.MarketDataSet, result.Request.Request.StaticDataSet, algo_id)
            result.Request.PreviousRequest.Model.MarketDataSet = market.to_canonical_format(result.Request.PreviousRequest.Model.MarketDataSet, result.Request.PreviousRequest.StaticDataSet, algo_id)
    env.logger.info('Converting done')

    env.logger.info('Collecting dependencies')

    request = ps.new_cm([])
    prev_request = ps.new_cm([])
    for res in results:
        if res['_NeedsToBeCalculated']:
            res['MarketDependencies'] = {}
            req = ps.get(ps.put(res.Request.Request))
            req.Queries = ['MarketDependencies']
            request.append(req)
            req = ps.get(ps.put(res.Request.PreviousRequest))
            req.Queries = ['MarketDependencies']
            prev_request.append(req)

    ps.put(request)
    ps.put(prev_request)
    env.logger.info(f'Requests collected. Request={request._id}. PrevRequest={prev_request._id}')

    res_market_dependencies = ps.compute_raw(algo_id, request).Result
    res_prev_market_dependencies = ps.compute_raw(algo_id, prev_request).Result

    all_market_dependencies = {}
    excluded_dependencies = {
        'RatesCurvesBundles': ['RUB_SOFR', 'USD_SOFR']
    }
    __update_marketDependencies(res_market_dependencies, results, all_market_dependencies, excluded_dependencies)
    __update_marketDependencies(res_prev_market_dependencies, results, all_market_dependencies, excluded_dependencies)

    del res_market_dependencies
    del res_prev_market_dependencies

    env.logger.info('Market dependencies collected')

    env.logger.info('Preparations done')

    # Calculating StaticDataSet change effect
    env.logger.info('Calculating StaticDataSet change effect')
    request = ps.new_cm([])
    prev_request = ps.new_cm([])
    needs_to_be_calculated = False
    for res in results:
        if res['_NeedsToBeCalculated']:
            req = res.Request
            if res['_NeedsToBeCalculated'] and req.Request.StaticDataSet._id != req.PreviousRequest.StaticDataSet._id:
                prev_request.append(req.PreviousRequest)
                new_req = ps.get(ps.put(req.PreviousRequest))
                new_req.StaticDataSet = req.Request.StaticDataSet
                request.append(new_req)
                needs_to_be_calculated = True
    if needs_to_be_calculated:
        __calculate_pnl_category(request, prev_request, results, algo_id, 'StaticDataSet')
        env.logger.info('StaticDataSet change calculated')
    else:
        env.logger.info('StaticDataSet change not needed')

    # Calculating Product change effect
    env.logger.info('Calculating Product change effect')
    request = ps.new_cm([])
    prev_request = ps.new_cm([])
    needs_to_be_calculated = False
    for res in results:
        if res['_NeedsToBeCalculated']:
            req = res.Request
            ns_from = ps.put(ps.new_cm(req.PreviousRequest.Product.NettingSets))
            ns_to = ps.put(ps.new_cm(req.Request.Product.NettingSets))
            if res['_NeedsToBeCalculated'] and ns_from._id != ns_to._id:
                prev_request.append(req.PreviousRequest)
                new_req = ps.get(ps.put(req.PreviousRequest))
                new_req.Product.NettingSets = req.Request.Product.NettingSets
                request.append(new_req)
                needs_to_be_calculated = True
                continue
    if needs_to_be_calculated:
        __calculate_pnl_category(request, prev_request, results, algo_id, 'Product')
        env.logger.info('Product change calculated')
    else:
        env.logger.info('Product change not needed')

    # Check if cpties were switched from credit curve to internal rating or vice versa
    env.logger.info('Credit model change')
    request = ps.new_cm([])
    prev_request = ps.new_cm([])
    needs_to_be_calculated = False
    for res in results:
        if res['_NeedsToBeCalculated']:
            req = res.Request
            if 'CounterpartyCreditRating' in req.Request.Product and 'CounterpartyCreditRating' not in req.PreviousRequest.Product:
                prev_request.append(req.PreviousRequest)
                new_req = ps.get(ps.put(req.PreviousRequest))
                del new_req.Product['CounterpartyCreditCurveIdentifier']
                new_req.Product['CounterpartyCreditRating'] = req.Request.Product.CounterpartyCreditRating
                request.append(new_req)
                needs_to_be_calculated = True
            if 'CounterpartyCreditRating' not in req.Request.Product and 'CounterpartyCreditRating' in req.PreviousRequest.Product:
                prev_request.append(req.PreviousRequest)
                new_req = ps.get(ps.put(req.PreviousRequest))
                del new_req.Product['CounterpartyCreditRating']
                new_req.Product['CounterpartyCreditCurveIdentifier'] = req.Request.Product.OwnCreditCurveIdentifier
                request.append(new_req)
                needs_to_be_calculated = True
    if needs_to_be_calculated:
        ps.put(request)
        ps.put(prev_request)
        env.logger.info(f'Requests collected. Request={request._id}. PrevRequest={prev_request._id}')
        __calculate_pnl_category(request, prev_request, results, algo_id, 'CreditModel')
        env.logger.info('Credit model change calculated')
    else:
        env.logger.info('Credit model change not needed')

    # Calculating Rating change effect
    env.logger.info('Calculating Rating change effect')
    request = ps.new_cm([])
    prev_request = ps.new_cm([])
    needs_to_be_calculated = False
    for res in results:
        if res['_NeedsToBeCalculated']:
            req = res.Request
            # If a cpty is calculated on its own curve, than we miss it
            if 'CounterpartyCreditRating' not in req.Request.Product or 'CounterpartyCreditRating' not in req.PreviousRequest.Product:
                continue
            if req.Request.Product.CounterpartyCreditRating != req.PreviousRequest.Product.CounterpartyCreditRating:
                prev_request.append(req.PreviousRequest)
                new_req = ps.get(ps.put(req.PreviousRequest))
                new_req.Product.CounterpartyCreditRating = req.Request.Product.CounterpartyCreditRating
                request.append(new_req)
                needs_to_be_calculated = True
    if needs_to_be_calculated:
        __calculate_pnl_category(request, prev_request, results, algo_id, 'Rating')
        env.logger.info('Rating change calculated')
    else:
        env.logger.info('Rating change not needed')

    # Calculating LGD change effect
    env.logger.info('Calculating LGD change effect')
    request = ps.new_cm([])
    prev_request = ps.new_cm([])
    needs_to_be_calculated = False
    for res in results:
        if res['_NeedsToBeCalculated']:
            req = res.Request
            if req.Request.Product.CounterpartyLossGivenDefault != req.PreviousRequest.Product.CounterpartyLossGivenDefault:
                prev_request.append(req.PreviousRequest)
                new_req = ps.get(ps.put(req.PreviousRequest))
                new_req.Product.CounterpartyLossGivenDefault = req.Request.Product.CounterpartyLossGivenDefault
                request.append(new_req)
                needs_to_be_calculated = True
    if needs_to_be_calculated:
        __calculate_pnl_category(request, prev_request, results, algo_id, 'LGD')
        env.logger.info('LGD change calculated')
    else:
        env.logger.info('LGD change not needed')

   # Model change
    env.logger.info('Model change effect calculation')
    for model_key in ['NumeraireCurrency', 'TypeName']:

        # First lets check model name and numeraire ccy change
        env.logger.info(model_key + ' change effect calculation')
        request = ps.new_cm([])
        prev_request = ps.new_cm([])
        needs_to_be_calculated = False
        for res in results:
            if res['_NeedsToBeCalculated']:
                req = res.Request
                if req.Request.Model[model_key] != req.PreviousRequest.Model[model_key]:
                    prev_request.append(req.PreviousRequest)
                    new_req = ps.get(ps.put(req.PreviousRequest))
                    new_req.Model[model_key] = req.Request.Model[model_key]
                    request.append(new_req)
                    needs_to_be_calculated = True
        if needs_to_be_calculated:
            __calculate_pnl_category(request, prev_request, result, algo_id, 'Model.' + model_key)
            env.logger.info(model_key + ' change effect done')
        else:
            env.logger.info(model_key + ' change not needed')

    # Theta
    env.logger.info('Theta effect calculation')
    request = []
    prev_request = ps.new_cm([])
    needs_to_be_calculated = False

    if same_mds:
        fixings_from = ps.new_cm(mds_from.Fixings)
        ps.put(fixings_from)
        fixings_to = ps.new_cm(mds_to.Fixings)
        ps.put(fixings_to)

        if mds_to.AsOfDate != mds_from.AsOfDate or fixings_to._id != fixings_from._id:
            request = ps.new_cm([x.Request.PreviousRequest for x in results if x['_NeedsToBeCalculated']])
            request = ps.get(ps.put(request))
            for req in request:
                req.Model.MarketDataSet.AsOfDate = mds_to.AsOfDate
                req.Model.MarketDataSet.Fixings = mds_to.Fixings
            prev_request = ps.new_cm([x.Request.PreviousRequest for x in results if x['_NeedsToBeCalculated']])
            ps.put(prev_request)
            needs_to_be_calculated = True
    else:
        for res in results:
            if res['_NeedsToBeCalculated']:
                req = res.Request

                fixings_from = ps.new_cm(req.PreviousRequest.Model.MarketDataSet.Fixings)
                fixings_to = ps.new_cm(req.Request.Model.MarketDataSet.Fixings)
                ps.put(fixings_from)
                ps.put(fixings_to)

                if req.Request.Model.MarketDataSet.AsOfDate != req.PreviousRequest.Model.MarketDataSet.AsOfDate\
                                                    or fixings_from._id != fixings_to._id:
                    prev_request.append(req.PreviousRequest)
                    new_req = ps.get(ps.put(req.PreviousRequest))
                    new_req.Model.MarketDataSet.AsOfDate = req.Request.Model.MarketDataSet.AsOfDate
                    request.append(new_req)
                    needs_to_be_calculated = True
        if needs_to_be_calculated:
            request = ps.new_cm(request)
            ps.put(request)
            prev_request = ps.new_cm(prev_request)
            ps.put(prev_request)
    if needs_to_be_calculated:
        env.logger.info(f'Requests collected. Request={request._id}. PrevRequest={prev_request._id}')
        __calculate_pnl_category(ps.new_cm(request), ps.new_cm(prev_request), results, algo_id, 'Model.MarketDataSet.AsOfDate')
        env.logger.info('Theta effect done')
    else:
        env.logger.info('Theta not needed')

    # Credit curves
    # Done separately for optimization purposes
    if optimize_credit_curves and all_market_dependencies.get('CreditCurves'):
        env.logger.info('Processing CreditCurves')
        request = []
        prev_request = []
        needs_to_be_calculated = False

        env.logger.info('Collecting requests')
        if same_mds:
            md_to = next(filter(lambda x: x.Request.get('Request') is not None, results)).Request.Request.Model.MarketDataSet
            md_from = next(filter(lambda x: x.Request.get('PreviousRequest') is not None, results)).Request.PreviousRequest.Model.MarketDataSet

            # for curve in all_market_dependencies['CreditCurves']:
            #     if md_to['CreditCurves'][curve]._id != md_from['CreditCurves'][curve]._id:
            #         needs_to_be_calculated = True
            #         break

            new_md_to = ps.get(ps.put(md_from))
            for curve in all_market_dependencies['CreditCurves']:
                new_md_to.CreditCurves[curve] = md_to.CreditCurves[curve]

            for res in results:
                req = res.Request
                if res['_NeedsToBeCalculated'] and 'CreditCurves' in res.MarketDependencies:
                    prev_request.append(req.PreviousRequest)
                    new_req = ps.get(ps.put(req.PreviousRequest))
                    new_req.Model.MarketDataSet = new_md_to
                    request.append(new_req)
                    needs_to_be_calculated = True
        else:
            for res in results:
                req = res.Request
                if res['_NeedsToBeCalculated'] and 'CresitCurves' in res.MarketDependencies:
                    md_to = req.Request.Model.MarketDataSet.CreditCurves
                    md_from = req.PreviousRequest.Model.MarketDataSet.CreditCurves

                    if md_to._id == md_from._id:
                        continue

                prev_request.append(req.PreviousRequest)
                new_req = ps.get(ps.put(req.PreviousRequest))
                new_req.Model.MarketDataSet.CreditCurves = md_to
                request.append(new_req)
                needs_to_be_calculated = True

        if needs_to_be_calculated:
            request = ps.new_cm(request)
            ps.put(request)
            prev_request = ps.new_cm(prev_request)
            ps.put(prev_request)
            env.logger.info(f'Requests collected. Request={request._id}. PrevRequest={prev_request._id}')

            # Calculating and removing unsuccessful results
            calc_request = ps.compute_raw(algo_id, request).Result
            calc_request = [x for x in calc_request if x.get('StatusText') is not None and x.StatusText == 'done']
            calc_prev_request = ps.compute_raw(algo_id, prev_request).Result
            calc_prev_request = [x for x in calc_prev_request if x.get('StatusText') is not None and x.StatusText == 'done']

            # Matching results
            for res in calc_request:
                cpty = res.Request['_Counterparty']
                result = next((x for x in results if x.Request['_Counterparty'] == cpty))
                prev_res = next((x for x in calc_prev_request if x.Request['_Counterparty'] == cpty), None)

                cva_change = res.Result['CVA'] - prev_res.Result['CVA'] if prev_res else None
                cva_curve = res.Request.Product.get('CounterpartyCreditCurveIdentifier') or \
                            res.Request.StaticDataSet.CreditRatingToSpreadMapping[str(res.Request.Product.CounterpartyCreditRating)].BaseCurveIdentifier
                dva_change = res.Result['DVA'] - prev_res.Result['DVA'] if prev_res else None
                dva_curve = res.Request.Product.get('OwnCreditCurveIdentifier')

                if 'ProductValue' in res.Request.Queries:
                    result['ProductValue'][f'Model.MarketDataSet.CreditCurves.{cva_curve}'] = 0
                    result['ProductValue'][f'Model.MarketDataSet.CreditCurves.{dva_curve}'] = 0
                if 'CVA' in res.Request.Queries:
                    result['CVA'][f'Model.MarketDataSet.CreditCurves.{cva_curve}'] = cva_change
                    result['CVA'][f'Model.MarketDataSet.CreditCurves.{dva_curve}'] = 0
                if 'DVA' in res.Request.Queries:
                    result['DVA'][f'Model.MarketDataSet.CreditCurves.{cva_curve}'] = 0
                    result['DVA'][f'Model.MarketDataSet.CreditCurves.{dva_curve}'] = dva_change
                if 'BCVA' in res.Request.Queries:
                    result['BCVA'][f'Model.MarketDataSet.CreditCurves.{cva_curve}'] = cva_change
                    if f'Model.MarketDataSet.CreditCurves.{dva_curve}' not in result['BCVA']:
                        result['BCVA'][f'Model.MarketDataSet.CreditCurves.{dva_curve}'] = 0
                    result['BCVA'][f'Model.MarketDataSet.CreditCurves.{dva_curve}'] += dva_change

                if prev_res:
                    calc_prev_request.remove(prev_res)

            for res in calc_prev_request:
                cpty = res.Request['_Counterparty']
                result = next((x for x in results if x.Request['_Counterparty'] == cpty))

                cva_curve = res.Request.Product.get('CounterpartyCreditCurveIdentifier') or \
                            res.Request.StaticDataSet.CreditRatingToSpreadMapping[str(res.Request.Product.CounterpartyCreditRating)].BaseCurveIdentifier
                dva_curve = res.Request.Product.get('OwnCreditCurveIdentifier')

                result['CVA'][f'Model.MarketDataSet.CreditCurves.{cva_curve}'] = 0
                result['CVA'][f'Model.MarketDataSet.CreditCurves.{dva_curve}'] = 0
                result['DVA'][f'Model.MarketDataSet.CreditCurves.{cva_curve}'] = 0
                result['DVA'][f'Model.MarketDataSet.CreditCurves.{dva_curve}'] = 0
                result['BCVA'][f'Model.MarketDataSet.CreditCurves.{cva_curve}'] = 0
                result['BCVA'][f'Model.MarketDataSet.CreditCurves.{dva_curve}'] = 0


        all_market_dependencies.pop('CreditCurves')

    # Other market data
    for Type in all_market_dependencies:
        for Identifier in all_market_dependencies[Type]:
            env.logger.info(f'Processing {Type}.{Identifier}')
            request = []
            prev_request = []
            needs_to_be_calculated = False

            env.logger.info('Collecting requests')
            if same_mds:

                md_to = next(filter(lambda x: x.Request.get('Request') is not None, results)).Request.Request.Model.MarketDataSet
                md_from = next(filter(lambda x: x.Request.get('PreviousRequest') is not None, results)).Request.PreviousRequest.Model.MarketDataSet

                if isinstance(md_to[Type][Identifier], float):
                    if md_to[Type][Identifier] == md_from[Type][Identifier]:
                        continue
                elif md_to[Type][Identifier]._id == md_from[Type][Identifier]._id:
                    continue

                new_md_to = ps.get(ps.put(md_from))
                new_md_to[Type][Identifier] = md_to[Type][Identifier]

                for res in results:
                    req = res.Request
                    if res['_NeedsToBeCalculated'] and Type in res.MarketDependencies \
                            and Identifier in res.MarketDependencies[Type]:

                        prev_request.append(req.PreviousRequest)
                        new_req = ps.get(ps.put(req.PreviousRequest))
                        new_req.Model.MarketDataSet = new_md_to
                        request.append(new_req)
                        needs_to_be_calculated = True
            else:
                for res in results:
                    req = res.Request
                    if res['_NeedsToBeCalculated'] and Type in res.MarketDependencies \
                            and Identifier in res.MarketDependencies[Type]:

                        md_to = req.Request.Model.MarketDataSet[Type][Identifier]
                        md_from = req.PreviousRequest.Model.MarketDataSet[Type][Identifier]

                        if isinstance(md_to, float):
                            if md_to == md_from:
                                continue
                        elif md_to._id == md_from._id:
                            continue

                        prev_request.append(req.PreviousRequest)
                        new_req = ps.get(ps.put(req.PreviousRequest))
                        new_req.Model.MarketDataSet[Type][Identifier] = req.Request.Model.MarketDataSet[Type][Identifier]
                        request.append(new_req)
                        needs_to_be_calculated = True

            if needs_to_be_calculated:
                request = ps.new_cm(request)
                ps.put(request)
                prev_request = ps.new_cm(prev_request)
                ps.put(prev_request)
                env.logger.info(f'Requests collected. Request={request._id}. PrevRequest={prev_request._id}')
                __calculate_pnl_category(request, prev_request, results, algo_id, 'Model.MarketDataSet.' + Type + '.' + Identifier)
                env.logger.info(f'Processing {Type}.{Identifier} done')
            else:
                env.logger.info(f'Processing {Type}.{Identifier} not needed')

    env.logger.info('Removing unnecessary info from result and adding Unexplained')
    for res in results:
        res.pop('MarketDependencies', None)
        res.pop('_NeedsToBeCalculated', None)
        res.Request.pop('Request', None)
        res.Request.pop('PreviousRequest', None)
        for query in res.Request.Queries:
            res[query]['Unexplained'] = 2*res[query]['Total'] - sum(res[query].values())

    env.logger.info('Unnecessary info deleted. Job is done')
    return ps.new_cm(results)


algo_id = sys.argv[3]
prev_response = ps.new_cm(ps.get(sys.argv[1]).Result)
response = ps.new_cm(ps.get(sys.argv[2]).Result)


t_0 = prev_response[0].Request.Model.MarketDataSet.AsOfDate
t_1 = response[0].Request.Model.MarketDataSet.AsOfDate

queries = ['BCVA', 'CVA', 'DVA']
big_request = prepare_big_request_for_pnl(response, prev_response,
                                                   algo_id,
                                                   'USD', queries)
a = run_generic_revaluation_based_metric_change_explain_batch(big_request, algo_id, True, -1, True)


ps_utils.env.logger.info('Saving results to Excel')
df = pd.DataFrame(columns=['MetricType', 'Category', 'Counterparty', 'Value'])
for result in a:
    for metric in queries:
        for category in result[metric]:
            df = pd.concat([df, pd.DataFrame([{
                'MetricType': metric,
                'Category': category,
                'Counterparty': result['_Counterparty'],
                'Value': result[metric][category]}])])

df_aggr = pd.DataFrame(df.groupby(['MetricType', 'Category'])['Value'].apply(sum)).pivot_table(index='Category', columns=['MetricType'], values='Value').fillna(0)

writer = pd.ExcelWriter(f'PnL Explain_{t_0.strftime("%Y-%m-%d")}_{t_1.strftime("%Y-%m-%d")}.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='PnLExplain', index=False)
df_aggr.to_excel(writer, sheet_name='PnLAggr', index=True)

wb = writer.book
ws1 = writer.sheets['PnLExplain']
ws2 = writer.sheets['PnLAggr']
nb_format = wb.add_format({'num_format': '#,##0'})
ws1.set_column(0, 0, 15)
ws1.set_column(1, 1, 60)
ws1.set_column(2, 2, 15)
ws1.set_column(3, 3, 10, nb_format)
ws2.set_column(0, 0, 60)
ws2.set_column(1, 3, 10, nb_format)

writer.close()
ps_utils.env.logger.info('Results saved!')

#ps.put(a)
#|ps_utils.env.logger.info(f'Result id is {a._id}')

