"""Utilities for PnL attribution
    """
from ps_utils.env import ps
import numpy as np
from ps_utils import env, market, analytics, pricing
import pandas as pd
from ps import message as msg

from typing import Union, Optional, Dict, List
PsObject = Union[ps.Message, ps.GlobalID]


def run_generic_revaluation_based_metric_change_explain(metrics_to_explain: List[str], reporting_ccy: str, pricing_request: pricing.GenericPricingRequest, previous_pricing_request: pricing.GenericPricingRequest, algo_id: str = None) -> PsObject:
    """Generates revaluation based explanation report for the metrics requested.

    Args:
        metrics_to_explain (List[str]): list of metrics
        reporting_ccy (str): currency of the report
        pricing_request (pricing.GenericPricingRequest): T pricing request
        previous_pricing_request (pricing.GenericPricingRequest): T-1 pricing request
        algo_id (str): ID of the algo to use

    Returns:
        PsObject: revaluation based PnL explain summary
    """

    if algo_id is None:
        algo_id = env.global_algo_id()

    # preparations
    pricing_request = pricing_request.clone()
    previous_pricing_request = previous_pricing_request.clone()

    pricing_request.set_reporting_currency(reporting_ccy)
    previous_pricing_request.set_reporting_currency(reporting_ccy)

    pricing_request.set_queries(metrics_to_explain)
    previous_pricing_request.set_queries(metrics_to_explain)

    # space for the output report
    report = {m: {} for m in metrics_to_explain}

    # base price
    r1 = previous_pricing_request.price()

    # final price
    r2 = pricing_request.price()
    for m in metrics_to_explain:
        report[m]['Total'] = r2[m] - r1[m]

    # let's convert market data to canonical format
    pricing_request.set_mds(market.to_canonical_format(pricing_request.get_mds(), pricing_request.get_sds(), algo_id))
    previous_pricing_request.set_mds(market.to_canonical_format(previous_pricing_request.get_mds(), previous_pricing_request.get_sds(), algo_id))

    if pricing_request.StaticDataSet._id != previous_pricing_request.StaticDataSet._id:
        # StaticDataSet has changed...
        sds = pricing_request.get_sds()
        r2 = previous_pricing_request.set_sds(sds).price()

        for m in metrics_to_explain:
            report[m]['StaticDataSet'] = r2[m] - r1[m]
        r1 = r2

    if pricing_request.Product._id != previous_pricing_request.Product._id:
        # Product has changed...
        r2 = previous_pricing_request.set_request_parameter('Product', pricing_request.Product).price()

        for m in metrics_to_explain:
            report[m]['Product'] = r2[m] - r1[m]
        r1 = r2

    if pricing_request.Model._id != previous_pricing_request.Model._id:
        # Model has changed...

        for model_key in ['NumeraireCurrency',  'TypeName']:
            if pricing_request.Model[model_key] != previous_pricing_request.Model[model_key]:
                # Model[model_key] has changed
                r2 = previous_pricing_request.set_model_parameter(model_key, pricing_request.Model[model_key]).price()

                for m in metrics_to_explain:
                    report[m][f'Model.{model_key}'] = r2[m] - r1[m]
                r1 = r2

        mds_from = previous_pricing_request.Model.MarketDataSet
        mds_to = pricing_request.Model.MarketDataSet

        if mds_from._id != mds_to._id:
            # Model.MarketDataSet has changed
            if mds_from.AsOfDate != mds_to.AsOfDate:
                mds_from.AsOfDate = mds_to.AsOfDate
                mds_from.Fixings = mds_to.Fixings

                previous_pricing_request.set_mds(mds_from)
                r2 = previous_pricing_request.price()

                for m in metrics_to_explain:
                    report[m][f'Model.MarketDataSet.AsOfDate'] = r2[m] - r1[m]
                r1 = r2

            deps = previous_pricing_request.get_market_dependencies()

            for Type in deps:
                for Identifier in deps[Type]:
                    env.logger.debug(f'Processing {Type} {Identifier}')

                    if Type not in mds_to:
                        mds_to[Type] = ps.new_cm()

                    if Type not in mds_from:
                        mds_from[Type] = ps.new_cm()

                    if Identifier not in mds_from[Type] and Identifier not in mds_to[Type]:
                        env.logger.debug(f'Skipping {Type} {Identifier} as it is not present in either market')
                        continue

                    if Identifier not in mds_to[Type]:
                        env.logger.debug(f'{Type} {Identifier} was dropped in upcoming market!')
                        mds_from[Type].pop(Identifier)
                    else:
                        if Identifier in mds_from[Type]:
                            if mds_from[Type][Identifier] == mds_to[Type][Identifier]:
                                env.logger.debug(f'Skipping {Type} {Identifier} as it did not move')
                                continue

                            if isinstance(mds_from[Type][Identifier], ps.Message) and mds_from[Type][Identifier]._id == mds_to[Type][Identifier]._id:
                                env.logger.debug(f'Skipping {Type} {Identifier} as it did not move')
                                continue

                            mds_from[Type][Identifier] = mds_to[Type][Identifier]

                    previous_pricing_request.set_mds(mds_from)
                    r2 = previous_pricing_request.price()

                    for m in metrics_to_explain:
                        report[m][f'Model.MarketDataSet.{Type}.{Identifier}'] = r2[m] - r1[m]
                    r1 = r2

    for m in metrics_to_explain:
        explained = 0
        for k in report[m]:
            if k == 'Total':
                continue

            if isinstance(report[m][k], dict):
                for k2 in report[m][k]:
                    explained += report[m][k][k2]
            else:
                explained += report[m][k]
        report[m]['Explained'] = explained
        report[m]['Unexplained'] = report[m]['Total'] - report[m]['Explained']

    msg = ps.put(ps.new_cm(report))
    env.logger.debug(f'Done revaluation based pnl explain: {msg}')
    return msg


def run_revaluation_based_pnl_explain(pricing_request: pricing.GenericPricingRequest, reporting_ccy: str, mds_from: PsObject, mds_to: PsObject, algo_id: str = None) -> PsObject:
    """Generates revaluation based PnL explain between two market data sets.

    Args:
        request (pricing.GenericPricingRequest): pricing request
        reporting_ccy (str): currency of the report
        mds_from (PsObject): market data set where pnl explain starts
        mds_to (PsObject): market data set where pnl explain ends
        algo_id (str): ID of the algo to use

    Returns:
        PsObject: revaluation based PnL explain summary
    """
    report = run_generic_revaluation_based_metric_change_explain(['ProductValue'],
                                                                 reporting_ccy,
                                                                 pricing_request.clone().set_mds(mds_to),
                                                                 pricing_request.clone().set_mds(mds_from),
                                                                 algo_id)

    return report['ProductValue']


def run_generic_risk_based_metric_change_explain(metric: str, reporting_ccy: str, pricing_request: pricing.GenericPricingRequest, mds_from: PsObject, mds_to: PsObject, algo_id: str = None) -> PsObject:
    """Generates risk based PnL explain between two market data sets.

    Args:
        metric (str): ProductValue, CVA or DVA
        reporting_ccy (str): currency of the report
        portfolio (pricing.GenericPricingRequest): pricing request
        mds_from (PsObject): market data set where pnl explain starts
        mds_to (PsObject): market data set where pnl explain ends
        algo_id (str): ID of the algo to use

    Returns:
        PsObject: revaluation based PnL explain summary
    """

    if algo_id is None:
        algo_id = env.global_algo_id()

    out = {}

    pricing_request = pricing_request.clone()
    pricing_request.set_reporting_currency(reporting_ccy)

    # Collect market dependencies
    env.logger.debug(f'Collecting market dependencies')
    deps = pricing_request.get_market_dependencies()

    # Trim market data sets
    env.logger.debug(f'Preparing market data for explanation')
    sds = pricing_request.get_sds()
    mds_from = market.trim_mds(market.to_canonical_format(mds_from, sds, algo_id), deps)
    mds_to = market.trim_mds(market.to_canonical_format(mds_to, sds, algo_id), deps)

    pfx = ''
    if metric in ['CVA', 'DVA']:
        pfx = f'{metric}_'
    elif metric != 'ProductValue':
        raise AttributeError(f'Cannot explain {metric} based on greeks. Can only dp ProductValue, CVA, DVA')

    # Product Value at T
    results_t = pricing_request.clone().set_mds(mds_from).price([metric, f'{pfx}Theta'])
    pv_t = results_t[metric]
    env.logger.debug(f'PV@T: {pv_t}')

    # Product Value at T+1
    results_t_1 = pricing_request.clone().set_mds(mds_to).price([metric])
    pv_t_1 = results_t_1[metric]

    env.logger.debug(f'PV@T+1: {pv_t_1}')

    env.logger.debug(f'Explaining PnL from {pv_t} to {pv_t_1}')
    total_pnl = pv_t_1 - pv_t

    out[f'{pfx}Theta'] = results_t[f'{pfx}Theta'] if mds_to.AsOfDate > mds_from.AsOfDate else 0.
    explained_pnl = out[f'{pfx}Theta']

    # Product Value at T@T+1
    # Advance from T to T+1
    if mds_to.AsOfDate > mds_from.AsOfDate:
        advanced_market_from = market.market_advance(mds_from, mds_to.AsOfDate, sds, algo_id)
        advanced_market_from.IsEOD = False
    else:
        advanced_market_from = mds_from

    results_t_at_t_1 = pricing_request.clone().set_mds(advanced_market_from).price([metric])
    pv_t_at_t_1 = results_t_at_t_1[metric]
    cashflows = results_t_at_t_1.CashToSettleToday if metric == 'ProductValue' else 0.
    env.logger.debug(f'PV@T->T+1 SOD: {pv_t_at_t_1}')

    # Collect PNL terms
    env.logger.debug(f'Collecting PNL terms')

    # ThetaPNL
    theta_pnl = out[f"{pfx}Theta"]
    env.logger.debug(f'Theta explains { theta_pnl / total_pnl * 100 }%')

    # CashFlows
    cash_pnl = -cashflows
    out["CashFlows"] = cash_pnl
    env.logger.debug(f'CashFlows explain { cash_pnl / total_pnl * 100 }%')
    explained_pnl += cash_pnl

    # Project advanced T market onto T+1 market benchmarks
    mds_projected_from = market.market_imply(mds_to, advanced_market_from, sds, algo_id)
    mds_projected_from.IsEOD = mds_to.IsEOD

    env.logger.debug(f'Calculating risk at advanced T market projected onto T+1 benchamrks')
    queries = [metric, f'{pfx}Delta', f'{pfx}Rho', f'{pfx}Vega']
    results = pricing_request.clone().set_mds(mds_projected_from).price(queries)

    # StructuralPNL
    pv_t_projected = results[metric]
    structural_changes_pnl = pv_t_projected - (pv_t_at_t_1 - cashflows)
    out['Structural'] = structural_changes_pnl
    explained_pnl += structural_changes_pnl
    env.logger.debug(f'Structural changes explain { structural_changes_pnl / total_pnl * 100 }%')

    # DeltaPNL
    if f'{pfx}Delta' in queries:
        out[f'{pfx}Delta'] = {}
        delta_pnl = 0
        for Identifier in set(deps.Spots):
            delta = results[f'{pfx}Delta'][Identifier]
            dSpot = mds_to.Spots[Identifier] - mds_projected_from.Spots[Identifier]
            pnl = delta * dSpot
            delta_pnl += pnl
            out[f'{pfx}Delta'][Identifier] = pnl
        explained_pnl += delta_pnl
        env.logger.debug(f'Delta explains { delta_pnl / total_pnl * 100 }%')

    # RhoPNL
    if f'{pfx}Rho' in queries:
        out[f'{pfx}Rho'] = {}
        rho_pnl = 0
        for Identifier in set(deps.RatesCurvesBundles):
            if Identifier in results[f'{pfx}Rho']:
                ir_delta = np.array(results[f'{pfx}Rho'][Identifier])

                rates_to = np.array(mds_to.RatesCurvesBundles[Identifier].RatesSchedule.Rates)
                if mds_to.RatesCurvesBundles[Identifier].RateQuoteExpression == 'Percentage':
                    rates_to *= 0.01

                rates_from = np.array(mds_projected_from.RatesCurvesBundles[Identifier].RatesSchedule.Rates)
                if mds_projected_from.RatesCurvesBundles[Identifier].RateQuoteExpression == 'Percentage':
                    rates_from *= 0.01

                drates = rates_to - rates_from

                offset = drates.shape[0] - ir_delta.shape[0]
                pnl_vector = ir_delta * drates[offset:]
                pnl = np.sum(pnl_vector)
                rho_pnl += pnl
                out[f'{pfx}Rho'][Identifier] = pnl_vector.tolist()
        explained_pnl += rho_pnl
        env.logger.debug(f'Rho explains { rho_pnl / total_pnl * 100 }%')

    # VegaPNL
    if f'{pfx}Vega' in queries:
        out[f'{pfx}Vega'] = {}
        vega_pnl = 0
        for Identifier in set(deps.VolatilitySurfaces):
            vega = np.array(results[f'{pfx}Vega'][Identifier])
            dVols = np.array(mds_to.VolatilitySurfaces[Identifier].VolatilityQuotes) - \
                np.array(mds_projected_from.VolatilitySurfaces[Identifier].VolatilityQuotes)

            offset = dVols.shape[0] - vega.shape[0]
            pnl_matrix = vega * dVols[offset:, :]
            pnl = np.sum(pnl_matrix)
            vega_pnl += pnl
            out[f'{pfx}Vega'][Identifier] = pnl_matrix.tolist()
        explained_pnl += vega_pnl
        env.logger.debug(f'Vega explains { vega_pnl / total_pnl* 100 }%')

    out['Total'] = pv_t_1 - pv_t
    out['Explained'] = explained_pnl
    out['Unexplained'] = out['Total'] - out['Explained']
    env.logger.debug(f'Risk explains { explained_pnl / total_pnl * 100 }%')

    msg = ps.put(ps.new_cm({metric: out}))
    env.logger.debug(f'Done risk based pnl explain for {mds_from.AsOfDate.date()}: {msg}')
    return msg


def run_risk_based_pnl_explain(pricing_request: pricing.GenericPricingRequest, reporting_ccy: str, mds_from: PsObject, mds_to: PsObject, algo_id: str = None) -> PsObject:
    """Generates risk based PnL explain between two market data sets.

    Args:
        portfolio (pricing.GenericPricingRequest): pricing request
        reporting_ccy (str): currency of the report
        mds_from (PsObject): market data set where pnl explain starts
        mds_to (PsObject): market data set where pnl explain ends
        algo_id (str): ID of the algo to use

    Returns:
        PsObject: revaluation based PnL explain summary
    """

    return run_generic_risk_based_metric_change_explain('ProductValue', reporting_ccy, pricing_request, mds_from, mds_to, algo_id)


def run_risk_based_pnl_backtest(request: pricing.GenericPricingRequest, markets, algo_id: str = None, reporting_ccy=None) -> PsObject:
    """Price portfolio on each market, analyse pnl attribution

    Args:
        request (PsObject): base request to work with
        date_from (datetime.date, optional): date from which to start pnl explain report
        date_till (datetime.date, optional): date till which to run pnl explain report. Defaults to datetime.date.today().
        algo_id (str): algo to use

    Returns:
        PsObject: with the report
    """

    if algo_id is None:
        algo_id = env.global_algo_id()

    deps = request.get_market_dependencies()
    clean_markets = [market.trim_mds(mds, deps) for mds in markets]

    out = {}
    for i, m in enumerate(clean_markets):

        if i == 0:
            T = m
            continue

        T_plus_1 = clean_markets[i]

        try:
            out[T_plus_1.AsOfDate] = run_risk_based_pnl_explain(request, reporting_ccy, T, T_plus_1, algo_id)
        except Exception as e:
            env.logger.error(f'On {T.AsOfDate}: -> {e}')
            out[T_plus_1.AsOfDate] = str(e)

        T = T_plus_1

    results = ps.new_cm({'Dates': list(out.keys()), 'PnlExplain': list(out.values())})
    env.logger.debug(f'Done risk based pnl backtest {ps.put(results)}')

    data = []
    index = []
    for date, pnl in out.items():
        entry = {}

        if isinstance(pnl, str):
            continue

        for key in pnl:
            entry[key] = analytics.aggragate_values(pnl[key])
        data.append(entry)
        index.append(date.date())

    df = pd.DataFrame(data, index=index)

    df['Unhedged.Cumulative'] = df['Total'].cumsum(axis=0)
    df['Unexplained.Cumulative'] = df['Unexplained'].cumsum(axis=0)
    df['Hedged.Delta.Cumulative'] = (df['Total'] - df['Structural'] - df['Theta'] - df['CashFlows'] - df['Delta']).cumsum(axis=0)
    df['Hedged.Delta.Rho.Cumulative'] = (df['Total'] - df['Structural'] - df['Theta'] - df['CashFlows'] - df['Delta'] - df['Rho']).cumsum(axis=0)
    df['Hedged.Delta.Rho.Vega.Cumulative'] = (df['Total'] - df['Structural'] - df['Theta'] - df['CashFlows'] -
                                              df['Delta'] - df['Rho']-df['Vega']).cumsum(axis=0)

    df['Hedged.Delta.Rho.Vega.CrossGamma.Cumulative'] = (df['Total'] - df['Structural'] - df['Theta'] - df['CashFlows'] - df['Delta'] -
                                                         df['CrossGammaMatrix'] - df['Rho']-df['Vega']).cumsum(axis=0)

    df['Hedged.Delta.Rho.Vega.CrossGamma.Divs.Cumulative'] = (df['Total'] - df['Structural']-df['Theta']-df['CashFlows']-df['Delta'] -
                                                              df['CrossGammaMatrix']-df['Rho']-df['Vega']-df['DividendsSensitivities']).cumsum(axis=0)

    df['Hedged.Delta.Rho.Vega.CrossGamma.Divs.Cega.Cumulative'] = (df['Total'] - df['Structural']-df['Theta']-df['CashFlows']-df['Delta'] -
                                                                   df['CrossGammaMatrix']-df['Rho']-df['Vega']-df['ParallelCega']-df['DividendsSensitivities']).cumsum(axis=0)

    return df


def prepare_big_request_for_pnl(batch: msg.MessageList, prev_batch: msg.MessageList, algo_id: str = None, reporting_currency: str = None, queries: List[str] = None) -> msg.MessageList:

    if algo_id is None:
        algo_id = env.global_algo_id()

    big_request = []

    for request in batch:
        cpty = request['_Counterparty']
        prev_request = next((x for x in prev_batch if x['_Counterparty'] == cpty), None)
        if prev_request is not None:
            big_request.append(ps.new_cm({
                'RequestName': 'Layer2',
                '_Counterparty': cpty,
                'Layer2Algo': algo_id,
                'ReportingCurrency': reporting_currency if reporting_currency is not None else request.ReportingCurrency,
                'Method': 'Revaluation',
                'Queries': queries if queries is not None else request.Queries,
                'Request': request,
                'PreviousRequest': prev_request
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
            result[query][category] = res.Result[query] - prev_res.Result[query] if prev_res is not None else 0
        if prev_res is not None:
            calc_prev_request.remove(prev_res)
            # TODO: in future add NewTrades here

    for res in calc_prev_request:
        cpty = res.Request['_Counterparty']
        result = next((x for x in results if x.Request['_Counterparty'] == cpty))
        for query in res.Request.Queries:
            result[query][category] = 0
        # TODO in future add ExpiredTrades here

def run_generic_revaluation_based_metric_change_explain_batch(big_request: msg.MessageList,
                                                              algo_id: str = None,
                                                              same_mds: bool = False) -> PsObject:
    """Generates revaluation based explanation report for the metrics requested.
    PS! PnL Explain is calculated from the same Base (compared to other methods)
    PPS! Make sure, input data is in correct format
    PPPS! One request per one cpty in big request!

    Args:
        big_request (MessageList): A list of by-counterparty collected requests on two compared dates
        algo_id (str): ID of the algo to use
        same_mds (bool): True if within date and within prev_date the same mds is assigned (improves performance)

    Returns:
        PsObject: revaluation based PnL explain summary
    """

    if algo_id is None:
        algo_id = env.global_algo_id()

    env.logger.info('Removing failed cpties')
    response = ps.compute_raw(algo_id, ps.new_cm([x.Request for x in big_request]))
    prev_response = ps.compute_raw(algo_id, ps.new_cm([x.PreviousRequest for x in big_request]))
    failed_cpties = {x.Request['_Counterparty'] for x in response.Result if x.get('StatusText') is None or x.StatusText != 'done'}
    failed_cpties.update({x.Request['_Counterparty'] for x in prev_response.Result if x.get('StatusText') is None or x.StatusText != 'done'})
    big_request = [x for x in big_request if x['_Counterparty'] not in failed_cpties]
    env.logger.info('Cpties removed')

    env.logger.info('PNL Explain started')

    env.logger.info('Doing some preparations')

    env.logger.info('Converting market data to canonical format')
    # Converting market data to canonical format
    if same_mds:
        mds_to = market.to_canonical_format(big_request[0].Request.Model.MarketDataSet)
        mds_from = market.to_canonical_format(big_request[0].PreviousRequest.Model.MarketDataSet)
        for request in big_request:
            request.Request.Model.MarketDataSet = mds_to
            request.PreviousRequest.Model.MarketDataSet = mds_from
    else:
        for request in big_request:
            request.Request.Model.MarketDataSet = market.to_canonical_format(big_request[0].Request.Model.MarketDataSet, big_request[0].Request.StaticDataSet, algo_id)
            request.PreviousRequest.Model.MarketDataSet = market.to_canonical_format(big_request[0].PreviousRequest.Model.MarketDataSet, big_request[0].PreviousRequest.StaticDataSet, algo_id)
    env.logger.info('Converting done')

    env.logger.info('Collecting dependencies')
    results = ps.new_cm([])
    all_market_dependencies = {}
    for req in big_request:
        req.Request.Queries = req.Queries
        req.PreviousRequest.Queries = req.Queries
        req.Request.ReportingCurrency = req.ReportingCurrency
        req.PreviousRequest.ReportingCurrency = req.ReportingCurrency

        req.MarketDependencies = pricing.XVA(req.Request).get_market_dependencies(algo_id)
        req.PreviousMarketDependencies = pricing.XVA(req.PreviousRequest).get_market_dependencies(algo_id)
        #TODO: refactor the code below
        for Type in req.MarketDependencies:
            for Identifier in req.MarketDependencies[Type]:
                md_to = req.Request.Model.MarketDataSet[Type][Identifier]
                md_from = req.PreviousRequest.Model.MarketDataSet[Type][Identifier]

                if isinstance(md_to, float):
                    if md_to != md_from:
                        if Type not in all_market_dependencies:
                            all_market_dependencies[Type] = []
                        if Identifier not in all_market_dependencies[Type]:
                            all_market_dependencies[Type] += [Identifier]
                elif md_to._id != md_from._id:
                    if Type not in all_market_dependencies:
                        all_market_dependencies[Type] = []
                    if Identifier not in all_market_dependencies[Type]:
                        all_market_dependencies[Type] += [Identifier]
        for Type in req.PreviousMarketDependencies:
            for Identifier in req.PreviousMarketDependencies[Type]:
                md_to = req.Request.Model.MarketDataSet[Type][Identifier]
                md_from = req.PreviousRequest.Model.MarketDataSet[Type][Identifier]

                if isinstance(md_to, float):
                    if md_to != md_from:
                        if Type not in all_market_dependencies:
                            all_market_dependencies[Type] = []
                        if Identifier not in all_market_dependencies[Type]:
                            all_market_dependencies[Type] += [Identifier]
                elif md_to._id != md_from._id:
                    if Type not in all_market_dependencies:
                        all_market_dependencies[Type] = []
                    if Identifier not in all_market_dependencies[Type]:
                        all_market_dependencies[Type] += [Identifier]

        result = ps.new_cm({'Request': req})
        for query in req.Queries:
            result[query] = ps.new_cm({})
        results.append(result)

    env.logger.info('Market dependencies collected')

    env.logger.info('Preparations done')

    # Calculating Total
    env.logger.info('Calculating Total')
    request = ps.new_cm([x.Request for x in big_request])
    prev_request = ps.new_cm([x.PreviousRequest for x in big_request])
    __calculate_pnl_category(request, prev_request, results, algo_id, 'Total')
    env.logger.info('Total calculated')

    # Calculating StaticDataSet change effect
    env.logger.info('Calculating StaticDataSet change effect')
    request = []
    prev_request = []
    needs_to_be_calculated = False
    for req in big_request:
        if req.Request.StaticDataSet._id != req.PreviousRequest.StaticDataSet._id:
            prev_request.append(req.PreviousRequest)
            new_req = ps.get(ps.put(req.PreviousRequest))
            new_req.StaticDataSet = req.Request.StaticDataSet
            request.append(new_req)
            needs_to_be_calculated = True
    if needs_to_be_calculated:
        __calculate_pnl_category(ps.new_cm(request), ps.new_cm(prev_request), results, algo_id, 'StaticDataSet')
        env.logger.info('StaticDataSet change calculated')
    else:
        env.logger.info('StaticDataSet change not needed')

    # Calculating Product change effect
    env.logger.info('Calculating Product change effect')
    request = []
    prev_request = []
    needs_to_be_calculated = False
    for req in big_request:
        if req.Request.Product._id != req.PreviousRequest.Product._id:
            prev_request.append(req.PreviousRequest)
            new_req = ps.get(ps.put(req.PreviousRequest))
            new_req.Product = req.Request.Product
            request.append(new_req)
            needs_to_be_calculated = True
    if needs_to_be_calculated:
        __calculate_pnl_category(ps.new_cm(request), ps.new_cm(prev_request), results, algo_id, 'Product')
        env.logger.info('Product change calculated')
    else:
        env.logger.info('Product change not needed')

    # Model change
    env.logger.info('Model change effect calculation')
    for model_key in ['NumeraireCurrency', 'TypeName']:

        # First lets check model name and numeraire ccy change
        env.logger.info(model_key + ' change effect calculation')
        request = []
        prev_request = []
        needs_to_be_calculated = False
        for req in big_request:
            if req.Request.Model[model_key] != req.PreviousRequest.Model[model_key]:
                prev_request.append(req.PreviousRequest)
                new_req = ps.get(ps.put(req.PreviousRequest))
                new_req.Model[model_key] = req.Request.Model[model_key]
                request.append(new_req)
                needs_to_be_calculated = True
        if needs_to_be_calculated:
            __calculate_pnl_category(ps.new_cm(request), ps.new_cm(prev_request), result, algo_id, 'Model.' + model_key)
            env.logger.info(model_key + ' change effect done')
        else:
            env.logger.info(model_key + ' change not needed')

    # Theta
    env.logger.info('Theta effect calculation')
    request = []
    prev_request = []
    needs_to_be_calculated = False

    if same_mds:
        if mds_to.AsOfDate != mds_from.AsOfDate:
            request = [ps.get(ps.put(x.PreviousRequest)) for x in big_request]
            for req in request:
                req.Model.MarketDataSet.AsOfDate = mds_to.AsOfDate
            prev_request = [x.PreviousRequest for x in big_request]
            needs_to_be_calculated = True
    else:
        for req in big_request:
            if req.Request.Model.MarketDataSet.AsOfDate != req.PreviousRequest.Model.MarketDataSet.AsOfDate:
                prev_request.append(req.PreviousRequest)
                new_req = ps.get(ps.put(req.PreviousRequest))
                new_req.Model.MarketDataSet.AsOfDate = req.Request.Model.MarketDataSet.AsOfDate
                request.append(new_req)
                needs_to_be_calculated = True
    if needs_to_be_calculated:
        __calculate_pnl_category(ps.new_cm(request), ps.new_cm(prev_request), results, algo_id, 'Model.MarketDataSet.AsOfDate')
        env.logger.info('Theta effect done')
    else:
        env.logger.info('Theta not needed')

    for Type in all_market_dependencies:
        for Identifier in all_market_dependencies[Type]:
            env.logger.info(f'Processing {Type}.{Identifier}')
            request = ps.new_cm([])
            prev_request = ps.new_cm([])
            needs_to_be_calculated = False

            env.logger.info('Collecting requests')
            if same_mds:
                md_to = big_request[0].Request.Model.MarketDataSet[Type][Identifier]
                md_from = big_request[0].PreviousRequest.Model.MarketDataSet[Type][Identifier]

                if isinstance(md_to, float):
                    if md_to == md_from:
                        continue
                elif md_to._id == md_from._id:
                    continue

                for req in big_request:
                    if Type in req.MarketDependencies \
                            and Identifier in req.MarketDependencies[Type] \
                            and Type in req.PreviousMarketDependencies \
                            and Identifier in req.PreviousMarketDependencies[Type]:

                        prev_request.append(req.PreviousRequest)
                        new_req = ps.get(ps.put(req.PreviousRequest))
                        new_req.Model.MarketDataSet[Type][Identifier] = req.Request.Model.MarketDataSet[Type][Identifier]
                        request.append(new_req)
                        needs_to_be_calculated = True
            else:
                for req in big_request:
                    if Type in req.MarketDependencies \
                            and Identifier in req.MarketDependencies[Type] \
                            and Type in req.PreviousMarketDependencies \
                            and Identifier in req.PreviousMarketDependencies[Type]:

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
        del res.Request['MarketDependencies']
        del res.Request['PreviousMarketDependencies']
        for query in res.Request.Queries:
            res[query]['Unexplained'] = 2*res[query]['Total'] - sum(res[query].values())

    env.logger.info('Unnecessary info deleted. Job is done')

    return ps.new_cm(results)

