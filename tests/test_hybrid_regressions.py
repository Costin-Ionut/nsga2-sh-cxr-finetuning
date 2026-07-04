from finetune_ga.core.training import fallback_attempts_for_backbone, choose_better_stage, best_epoch_by_auc


def test_fallback_attempts_monotonic_reduce_resources():
    attempts = fallback_attempts_for_backbone('resnet', batch_size=32, target_size=224, n_last_layers=16)
    assert attempts[0] == {'target_size': 224, 'batch_size': 32, 'n_last_layers': 16}
    assert any(a['batch_size'] < 32 for a in attempts)
    assert any(a['n_last_layers'] == 0 for a in attempts)
    assert any(a['target_size'] < 224 for a in attempts)


def test_best_stage_selection_prefers_higher_auc_then_lower_loss():
    s1 = (0.88, 0.30, 2)
    s2 = (0.90, 0.50, 1)
    assert choose_better_stage(s1, s2) == (0.90, 0.50, 1, 2)
    s3 = (0.90, 0.28, 1)
    assert choose_better_stage(s3, s2) == (0.90, 0.28, 1, 1)


def test_val_metrics_are_taken_from_winning_epoch():
    class FakeHistory:
        def __init__(self, auc, pr, acc, loss):
            self.history = {'val_auc': auc, 'val_pr_auc': pr, 'val_accuracy': acc, 'val_loss': loss}
    h1 = FakeHistory([0.80, 0.88, 0.91], [0.70, 0.78, 0.80], [0.82, 0.85, 0.87], [0.4, 0.3, 0.25])
    h2 = FakeHistory([0.85, 0.87], [0.88, 0.95], [0.84, 0.86], [0.28, 0.26])
    s1 = best_epoch_by_auc(h1)
    s2 = best_epoch_by_auc(h2)
    best_auc, best_loss, best_epoch, best_stage = choose_better_stage(s1, s2)
    assert best_stage == 1
    assert best_epoch == 2
    winning = h1 if best_stage == 1 else h2
    assert winning.history['val_pr_auc'][best_epoch] == 0.80
    assert winning.history['val_accuracy'][best_epoch] == 0.87
