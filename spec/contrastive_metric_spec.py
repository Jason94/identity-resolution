from mamba import *  # type: ignore
from expects import *  # type: ignore
from expects.matchers import Matcher
import torch
import math

from config import *
from contrastive_metric import ContrastiveLoss


class be_about(Matcher):
    def __init__(self, expected, precision=1e-5):
        self.expected = expected
        self.precision = precision

    def _match(self, actual):
        distance = abs(self.expected - actual)
        if distance < self.precision:
            return True, ["The values are close enough"]
        return False, [str(self.expected), f"but is {str(distance)} away"]


DUPLICATE_TARGET = torch.tensor([DUPLICATE_CLASS])
DISTINCT_TARGET = torch.tensor([DISTINCT_CLASS])

with description(ContrastiveLoss) as self:  # type: ignore
    with it("initializes correctly"):  # type: ignore
        contrastive_loss = ContrastiveLoss(margin=2.0)
        expect(contrastive_loss.margin).to(equal(2.0))

    with context("when forward method is used"):  # type: ignore
        with it("computes loss correctly for duplicate pairs"):  # type: ignore
            contrastive_loss = ContrastiveLoss(margin=2.0)
            output_1 = torch.tensor([1.0, 0.5, 2.5])
            output_2 = torch.tensor([-1.0, 0.2, 1.0])

            loss = contrastive_loss(output_1, output_2, DUPLICATE_TARGET)

            target = math.sqrt(
                (1.0 - (-1.0)) ** 2 + (0.5 - 0.2) ** 2 + (2.5 - 1.0) ** 2
            )

            expect(loss.item()).to(be_about(target))

        with it("computes loss correctly for distinct pairs within the margin"):  # type: ignore
            contrastive_loss = ContrastiveLoss(margin=2.0)
            output_1 = torch.tensor([0.0, 1.0, 0.0])
            output_2 = torch.tensor([1.0, 0.0, 1.0])

            loss = contrastive_loss(output_1, output_2, DISTINCT_TARGET)

            distance = math.sqrt(3)
            expect(distance).to(
                be_below(2.0)
            )  # Just to make sure we're within the margin!
            target = 2.0 - distance
            expect(loss.item()).to(be_about(target))

        with it("computes loss correctly for distinct pairs outside the margin"):  # type: ignore
            contrastive_loss = ContrastiveLoss(margin=2.0)
            output_1 = torch.tensor([0.0, 5.0, 0.0])
            output_2 = torch.tensor([5.0, 0.0, 5.0])

            loss = contrastive_loss(output_1, output_2, DISTINCT_TARGET)

            distance = math.sqrt(15)
            expect(distance).to(
                be_above(2.0)
            )  # Just to make sure we're outside the margin!

            expect(loss.item()).to(be_about(0.0))

        with context("when batching"):  # type: ignore
            with it("computes loss correctly for duplicate pairs"):  # type: ignore
                contrastive_loss = ContrastiveLoss(margin=2.0)
                output_1_1 = torch.tensor([1.0, 1.0, 0.0])
                output_1_2 = torch.tensor([1.0, 1.0, 1.0])

                output_2_1 = torch.tensor([0.1, 0.0, 0.0])
                output_2_2 = torch.tensor([0.5, 0.5, 0.25])

                loss = contrastive_loss(
                    torch.stack([output_1_1, output_2_1]),
                    torch.stack([output_1_2, output_2_2]),
                    torch.tensor([DUPLICATE_CLASS, DUPLICATE_CLASS]),
                )

                distance_1 = math.sqrt(1.0)
                distance_2 = math.sqrt((0.5 - 0.1) ** 2 + 0.5**2 + 0.25**2)
                target = (distance_1 + distance_2) / 2.0

                expect(loss.item()).to(be_about(target))

            with it("computes loss correctly for distinct pairs outside the margin"):  # type: ignore
                contrastive_loss = ContrastiveLoss(margin=2.0)
                output_1_1 = torch.tensor([0.0, 1.0, 0.0])
                output_1_2 = torch.tensor([4.0, 1.0, 4.0])

                output_2_1 = torch.tensor([-4.0, 0.0, -10.0])
                output_2_2 = torch.tensor([4.0, -8.0, -2.0])

                loss = contrastive_loss(
                    torch.stack([output_1_1, output_2_1]),
                    torch.stack([output_1_2, output_2_2]),
                    torch.tensor([DISTINCT_CLASS, DISTINCT_CLASS]),
                )

                distance_1 = math.sqrt(4.0**2 + 4.0**2)
                distance_2 = math.sqrt(8.0**2 + 8.0**2 + 8.0**2)
                expect(distance_1).to(be_above(2.0))
                expect(distance_2).to(be_above(2.0))

                expect(loss.item()).to(be_about(0.0))

            with it("computes loss correctly for distinct pairs inside the margin"):  # type: ignore
                contrastive_loss = ContrastiveLoss(margin=2.0)
                output_1_1 = torch.tensor([1.0, 1.0, 0.0])
                output_1_2 = torch.tensor([1.0, 1.0, 1.0])

                output_2_1 = torch.tensor([0.1, 0.0, 0.0])
                output_2_2 = torch.tensor([0.5, 0.5, 0.25])

                loss = contrastive_loss(
                    torch.stack([output_1_1, output_2_1]),
                    torch.stack([output_1_2, output_2_2]),
                    torch.tensor([DISTINCT_CLASS, DISTINCT_CLASS]),
                )

                distance_1 = math.sqrt(1.0)
                distance_2 = math.sqrt((0.5 - 0.1) ** 2 + 0.5**2 + 0.25**2)
                expect(distance_1).to(be_below(2.0))
                expect(distance_2).to(be_below(2.0))

                target_1 = 2.0 - distance_1
                target_2 = 2.0 - distance_2
                target = (target_1 + target_2) / 2.0

                expect(loss.item()).to(be_about(target))

            with it("computes loss correctly for distinct pairs in & out the margin"):  # type: ignore
                contrastive_loss = ContrastiveLoss(margin=2.0)
                output_1_1 = torch.tensor([1.0, 1.0, 0.0])
                output_1_2 = torch.tensor([1.0, 1.0, 1.0])

                output_2_1 = torch.tensor([-4.0, 0.0, -10.0])
                output_2_2 = torch.tensor([4.0, -8.0, -2.0])

                loss = contrastive_loss(
                    torch.stack([output_1_1, output_2_1]),
                    torch.stack([output_1_2, output_2_2]),
                    torch.tensor([DISTINCT_CLASS, DISTINCT_CLASS]),
                )

                distance_1 = math.sqrt(1.0)
                distance_2 = math.sqrt(8.0**2 + 8.0**2 + 8.0**2)
                expect(distance_1).to(be_below(2.0))
                expect(distance_2).to(be_above(2.0))

                target_1 = 2.0 - distance_1
                target_2 = 0.0
                target = (target_1 + target_2) / 2.0

                expect(loss.item()).to(be_about(target))

            with it("computes loss correctly for mixed pairs, distinct inside the margin"):  # type: ignore
                contrastive_loss = ContrastiveLoss(margin=2.0)
                output_1_1 = torch.tensor([1.0, 1.0, 0.0])
                output_1_2 = torch.tensor([1.0, 1.0, 1.0])

                output_2_1 = torch.tensor([0.1, 0.0, 0.0])
                output_2_2 = torch.tensor([0.5, 0.5, 0.25])

                loss = contrastive_loss(
                    torch.stack([output_1_1, output_2_1]),
                    torch.stack([output_1_2, output_2_2]),
                    torch.tensor([DUPLICATE_CLASS, DISTINCT_CLASS]),
                )

                distance_1 = math.sqrt(1.0)
                distance_2 = math.sqrt((0.5 - 0.1) ** 2 + 0.5**2 + 0.25**2)
                expect(distance_2).to(be_below(2.0))

                target_1 = distance_1
                target_2 = 2.0 - distance_2
                target = (target_1 + target_2) / 2.0

                expect(loss.item()).to(be_about(target))

            with it("computes loss correctly for mixed pairs, distinct outside the margin"):  # type: ignore
                contrastive_loss = ContrastiveLoss(margin=2.0)
                output_1_1 = torch.tensor([1.0, 1.0, 0.0])
                output_1_2 = torch.tensor([1.0, 1.0, 1.0])

                output_2_1 = torch.tensor([-4.0, 0.0, -10.0])
                output_2_2 = torch.tensor([4.0, -8.0, -2.0])

                loss = contrastive_loss(
                    torch.stack([output_1_1, output_2_1]),
                    torch.stack([output_1_2, output_2_2]),
                    torch.tensor([DUPLICATE_CLASS, DISTINCT_CLASS]),
                )

                distance_1 = math.sqrt(1.0)
                distance_2 = math.sqrt(8.0**2 + 8.0**2 + 8.0**2)
                expect(distance_2).to(be_above(2.0))

                target_1 = distance_1
                target_2 = 0.0
                target = (target_1 + target_2) / 2.0

                expect(loss.item()).to(be_about(target))
