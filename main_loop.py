import os

from data_helper import SimpleListDataset, BlkPosInterface
from introspector_module import IntrospectorModule
from reasoner_module import ReasonerModule

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def main_loop():
    train_source = os.path.join(root_dir, 'data', 'newsqa_train_roberta-base.pkl')
    qd_dataset = SimpleListDataset(train_source)
    interface = BlkPosInterface(qd_dataset)

    num_samples = '1,1,1,1'
    intro_dataset = interface.build_random_buffer(num_samples=num_samples)
    introspector = IntrospectorModule(train_data_size=len(intro_dataset), epochs=3, batch_size=2)
    introspector.training_fit(intro_dataset)

    interface.collect_estimations_from_dir()
    reason_dataset = interface.build_promising_buffer(num_samples=num_samples)
    reasoner = ReasonerModule(train_data_size=len(reason_dataset), epochs=3, batch_size=2)
    reasoner.training_fit(reason_dataset)

if __name__ == "__main__":
    main_loop()
