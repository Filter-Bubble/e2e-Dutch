import e2edutch.metrics
import math


def test_f1():
    num = 10
    p_den = 100
    r_den = 50
    f1 = e2edutch.metrics.f1(num, p_den, num, r_den)
    assert math.isclose(f1, 2.0/15)


def test_f1_zero():
    num = 0
    p_den = 0
    r_den = 0
    f1 = e2edutch.metrics.f1(num, p_den, num, r_den)
    assert math.isclose(f1, 0)


def test_corefevaluator():
    evaluator = e2edutch.metrics.CorefEvaluator()
    assert evaluator.get_f1() == 0
    assert evaluator.get_recall() == 0
    assert evaluator.get_precision() == 0

    # empty update
    evaluator.update([], [], {}, {})
    assert evaluator.get_f1() == 0
    assert evaluator.get_recall() == 0
    assert evaluator.get_precision() == 0


def test_evaluate_documents():
    metrics = [e2edutch.metrics.muc,
               e2edutch.metrics.b_cubed,
               e2edutch.metrics.lea,
               e2edutch.metrics.ceafe]
    documents = []
    for metric in metrics:
        p, r, f = e2edutch.metrics.evaluate_documents(documents, metric)
        assert p == 0
        assert r == 0
        assert f == 0


def example_clusters():
    mentions = tuple([(i*2, i*2+1) for i in range(9)])
    a, b, c, d, e, f, g, h, i = mentions
    pred_clusters = [(a, b), (c, d), (f, g, h, i)]
    gold_clusters = [(a, b, c), (d, e, f, g)]
    mentions_to_gold = {m: cl for cl in tuple(gold_clusters) for m in cl}
    return pred_clusters, gold_clusters, mentions_to_gold


def test_muc():
    clusters = [((0, 1), (2, 3))]
    mentions_to_gold = {m: cl for cl in tuple(clusters) for m in cl}
    num, dem = e2edutch.metrics.muc(clusters, mentions_to_gold)
    assert math.isclose(num/dem, 1)

    clusters, _, mentions_to_gold = example_clusters()
    num, dem = e2edutch.metrics.muc(clusters, mentions_to_gold)
    assert math.isclose(num/dem, 0.4)


def test_b_cubed():
    clusters = [((0, 1), (2, 3))]
    mentions_to_gold = {m: cl for cl in tuple(clusters) for m in cl}
    num, dem = e2edutch.metrics.b_cubed(clusters, mentions_to_gold)
    assert math.isclose(num/dem, 1)

    clusters, _, mentions_to_gold = example_clusters()
    num, dem = e2edutch.metrics.b_cubed(clusters, mentions_to_gold)
    assert math.isclose(num/dem, 0.5)


def test_ceafe():
    clusters = [((0, 1), (2, 3))]
    pn, pd, rn, rd = e2edutch.metrics.ceafe(clusters, clusters)
    assert math.isclose(pn/pd, 1)

    clusters, gold_clusters, _ = example_clusters()
    pn, pd, rn, rd = e2edutch.metrics.ceafe(clusters, gold_clusters)
    assert math.isclose(pn/pd, 0.43, abs_tol=0.005)
