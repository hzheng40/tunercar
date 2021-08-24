"""
Author
------
    neurotic amoeba
    T RDA SDT DRS-US
    Siemens Corporation, Technology
    Princeton, NJ - 08540
"""

from prob_design_generator.space import DesignSpace
import networkx as nx
import pickle as pk
import numpy as np


# tube params: [angle, length]
# hub params: [type, angle]
# flange params: []
# battery: [type, capacity, mass]
# esc: [type, mass]
# motor: [type, mass, KT, KV]
# aircraft: [mass, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, CGx, CGy, CGz]


class Design(object):
    def __init__(self, node_options, end_options):
        super().__init__()

        self._node_options = node_options
        self._end_options = end_options

        self.selections = None

        assert len(node_options) == len(end_options)

        self.nodes = []
        self.edges = []

    def generate(self):
        self.nodes.append("orient__0")
        self.nodes.append("plate__1")
        self.nodes.append("battery__2")
        base_node = np.random.choice(self._node_options)
        node_name = f"{base_node}__{len(self.nodes)}"
        self.nodes.append(node_name)
        self.edges.append((3, 0, "orientation", "Orient_Connector", "Mount"))
        self.edges.append((1, 3, "mechanical", "BottomConnector",
                           "TopMountInverse_Connector"))
        self.edges.append((2, 1, "mechanical", "Bottom_Connector",
                           "TopConnector_1"))
        self._add_hub(base_node, node_name, level=0)

    def generate_by_selections(self, selections):
        """
        Generate designs based on vector of selections instead of random choice

        Args:
            selections (numpy.ndarray(int): (num_selections, )): selected component at each step

        Returns:
            None
        """
        selections.reverse()
        base_node = self._node_options[selections.pop()]
        node_name = f"{base_node}__{len(self.nodes)}"
        assert self.selections is None
        self.selections = selections
        self.nodes.append("orient__0")
        self.nodes.append("plate__1")
        self.nodes.append("battery__2")
        self.nodes.append(node_name)
        self.edges.append((3, 0, "orientation", "Orient_Connector", "Mount"))
        self.edges.append((1, 3, "mechanical", "BottomConnector",
                           "TopMountInverse_Connector"))
        self.edges.append((2, 1, "mechanical", "Bottom_Connector",
                           "TopConnector_1"))
        self._add_hub(base_node, node_name, level=0)

    def to_design_graph(self, space):
        additional_edges = []
        prop_count = 0

        design_graph = nx.DiGraph()
        mapping = {}
        for node in self.nodes:
            node_type = node.split("__")[0]
            if node_type.startswith("hub_"):
                hubname = f"0394od_para_hub_{node_type.split('_')[-1]}"
                container = space.find("HubConnectors")
                variant = getattr(container, hubname)
                node_instance = container.instantiate_variant(variant)
                self._insert_node(node, design_graph, "HubConnectors",
                                  node_instance, mapping)

            elif node_type.startswith("orient"):
                container = space.find("Orientation")
                variant = container.variants[0]
                node_instance = container.instantiate_variant(variant)
                self._insert_node(node, design_graph, "Orientation",
                                  node_instance, mapping)

            elif node_type.startswith("battery"):
                container = space.find("Battery")
                variant = getattr(container, "TurnigyGraphene3000mAh3S75C")
                node_instance = container.instantiate_variant(variant)
                battery_name = self._insert_node(node, design_graph, "Battery",
                                                 node_instance, mapping)

            elif node_type.startswith("plate"):
                container = space.find("PlateConnectors")
                variant = getattr(container, "para_cf_fplate")
                node_instance = container.instantiate_variant(variant)
                node_instance.X1_OFFSET.default = "0"
                node_instance.Z1_OFFSET.default = "1"
                self._insert_node(node, design_graph, "PlateConnectors",
                                  node_instance, mapping)

            elif node_type.startswith("flange_"):
                container = space.find("FlangeConnectors")
                variant = getattr(container, "0394_para_flange")
                node_instance = container.instantiate_variant(variant)
                flange_name = self._insert_node(node, design_graph,
                                                "FlangeConnectors",
                                                node_instance, mapping)

                # add motor
                container = space.find("Motor")
                variant = getattr(container, "t_motor_AT2312KV1400")
                node_instance = container.instantiate_variant(variant)
                motor_name = self._insert_node(node_instance.name,
                                               design_graph, "Motor",
                                               node_instance, mapping)

                # add propeller
                container = space.find("Propeller")
                if prop_count % 2 == 0:
                    variant = getattr(container, "apc_propellers_10x7E")
                else:
                    variant = getattr(container, "apc_propellers_10x7EP")
                node_instance = container.instantiate_variant(variant)
                prop_name = self._insert_node(node_instance.name, design_graph,
                                              "Propeller", node_instance,
                                              mapping)

                # add ESC
                container = space.find("ESC")
                variant = getattr(container, "ESC_debugging")
                node_instance = container.instantiate_variant(variant)
                esc_name = self._insert_node(node_instance.name, design_graph,
                                             "ESC", node_instance, mapping)

                additional_edges.extend([
                    (battery_name, esc_name, "electrical", "PowerBus", "PowerBus"),   # noqa
                    (esc_name, motor_name, "electrical", "MotorPower", "MotorPower"),   # noqa
                    (motor_name, flange_name, "mechanical", "Base", "TopConnector"),   # noqa
                    (prop_name, motor_name, "mechanical", "motor_connector", "Power_Out"),   # noqa
                    (esc_name, flange_name, "mechanical", "Connector", "EscSideConnector"),   # noqa
                ])

            elif node_type.startswith("tube"):
                container = space.find("TubeConnectors")
                variant = getattr(container, "0394OD_para_tube")
                node_instance = container.instantiate_variant(variant)
                self._insert_node(node, design_graph, "TubeConnectors",
                                  node_instance, mapping)

            elif node_type.startswith("left"):
                # add left wing and servo
                container = space.find("LeftWing")
                variant = getattr(container, "left_NACA_2612")
                node_instance = container.instantiate_variant(variant)
                self._insert_node(node, design_graph, "LeftWing",
                                  node_instance, mapping)
                # wing_name = self._insert_node(node, design_graph, "LeftWing",
                #                               node_instance, mapping)

                # add servo? unnecessarily adds weight
                # container = space.find("Servo")
                # variant = getattr(container, "Hitec_D485HW")
                # node_instance = container.instantiate_variant(variant)
                # servo_name = self._insert_node(node_instance.name,
                #                                design_graph, "Servo",
                #                                node_instance, mapping)

                # additional_edges.extend([
                #     (servo_name, wing_name, "mechanical", "Connector", "Wing_Servo_Connector")  # noqa
                # ])

            else:
                # add right wing and servo
                container = space.find("RightWing")
                variant = getattr(container, "right_NACA_2612")
                node_instance = container.instantiate_variant(variant)
                self._insert_node(node, design_graph, "RightWing",
                                  node_instance, mapping)
                # wing_name = self._insert_node(node, design_graph,
                #                               "RightWing", node_instance,
                #                               mapping)

                # add servo? unnecessarily adds weight
                # container = space.find("Servo")
                # variant = getattr(container, "Hitec_D485HW")
                # node_instance = container.instantiate_variant(variant)
                # servo_name = self._insert_node(node_instance.name,
                #                                design_graph, "Servo",
                #                                node_instance, mapping)

                # additional_edges.extend([
                #     (servo_name, wing_name, "mechanical", "Connector", "Wing_Servo_Connector")  # noqa
                # ])

        for source, target, conntype, sourceport, targetport in self.edges:
            source_node = mapping[self.nodes[source]]
            target_node = mapping[self.nodes[target]]
            design_graph.add_edge(source_node, target_node,
                                  connection_type=conntype,
                                  source_port=sourceport,
                                  target_port=targetport)
        for source, target, conntype, sourceport, targetport in additional_edges:  # noqa
            design_graph.add_edge(source, target, connection_type=conntype,
                                  source_port=sourceport,
                                  target_port=targetport)

        return design_graph

    def _insert_node(self, node, graph, instance, component, mapping):
        node_name = f"{component.name}__{len(graph)}"
        mapping.update({node: node_name})
        graph.add_node(node_name, node=component, instance=instance)
        return node_name

    def _add_hub(self, node_type, node_name, level=0):
        for i in range(int(node_type.split("_")[-1])):
            tube_name = f"tube__{len(self.nodes)}"
            self.nodes.append(tube_name)
            source = self.nodes.index(node_name)
            target = self.nodes.index(tube_name)
            self.edges.append((target, source, "mechanical", "EndConnection_1",
                               f"Connector_{i + 1}"))
            self._grow(tube_name, level + 1)

    def _grow(self, tube_name, level=0):
        if len(self.selections) == 0:
            return
        if self.selections is not None:
            if level <= 2:
                chosen = self._node_options[self.selections.pop()]
            else:
                chosen = self._end_options[self.selections.pop()]
        else:
            if level <= 2:
                options = self._node_options + self._end_options
                weights = ([1] * len(self._node_options) + [6, 3, 6, 0, 0])
                weights = [_ / sum(weights) for _ in weights]
            else:
                options = self._end_options
                weights = [6, 3, 6, 0, 0]
                weights = [_ / sum(weights) for _ in weights]
            chosen = np.random.choice(options, p=weights)

        chosen_name = f"{chosen}__{len(self.nodes)}"
        self.nodes.append(chosen_name)
        source = self.nodes.index(tube_name)
        target = self.nodes.index(chosen_name)
        if chosen.startswith("hub_"):
            target_port = "Connector_1"
        elif chosen.startswith("flange_side"):
            target_port = "SideConnector"
        elif chosen == "flange_bottom":
            target_port = "BottomConnector"
        elif chosen.endswith("wing"):
            target_port = "Wing_Tube_Connector"
        else:
            print(chosen)
            raise ValueError("Not supported.")
        self.edges.append((target, source, "mechanical", target_port,
                           "EndConnection_2"))

        if chosen.startswith("hub_"):
            self._add_hub(chosen, chosen_name, level + 1)
        else:
            if chosen == "flange_side_2":
                # get another go at adding something at the same level.
                self._grow(tube_name, level)


if __name__ == "__main__":
    design_space_filename = ("/home/tunercar/swri-uav-pipeline/swri-uav-exploration/assets/uav_design_space.acel")
    space = DesignSpace(design_space_filename)

    node_options = ["hub_2", "hub_3", "hub_4", "hub_5", "hub_6"]
    end_options = ["flange_side", "flange_side_2", "flange_bottom",
                   "left_wing", "right_wing"]
    design = Design(node_options, end_options)
    design.generate_by_selections([1,2,3,4,3,2,1,2,3,4,3,2,1,2,3,4])
    design_graph = design.to_design_graph(space)
    with open("./design_graph.pk", "wb") as fout:
        pk.dump(design_graph, fout)
