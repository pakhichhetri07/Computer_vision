model.load_state_dict(best_model_wts)
    test_acc, test_f1, test_auc = evaluate(model, test_loader)
    print(f"Fold {fold+1} Test: Accuracy={test_acc:.4f}, F1={test_f1:.4f}, AUROC={test_auc:.4f}")
    all_test_metrics.append((test_acc, test_f1, test_auc))

# Average Test Performance
avg_acc = np.mean([m[0] for m in all_test_metrics])
avg_f1 = np.mean([m[1] for m in all_test_metrics])
avg_auc = np.mean([m[2] for m in all_test_metrics])

print(f"\n Average Test Performance across 5 folds:")
print(f"Accuracy: {avg_acc:.4f}, F1-Score: {avg_f1:.4f}, AUROC: {avg_auc:.4f}")
